use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::{cmma, prelude::*};
use half::{bf16, f16};

use crate::matmul::matrix_layout::{as_cmma_layout, MatrixLayout};
use crate::matmul::MatmulInstruction;

pub struct CmmaInstruction16_16_16<I: Numeric, O: Numeric> {
    _input: PhantomData<I>,
    _output: PhantomData<O>,
}

pub struct CmmaInstruction32_8_16<I: Numeric, O: Numeric> {
    _input: PhantomData<I>,
    _output: PhantomData<O>,
}

pub struct CmmaInstruction8_32_16<I: Numeric, O: Numeric> {
    _input: PhantomData<I>,
    _output: PhantomData<O>,
}

pub trait CmmaValid<I: Numeric, O: Numeric> {}
impl CmmaValid<f16, f16> for (f16, f16) {}
impl CmmaValid<f16, f32> for (f16, f32) {}
impl CmmaValid<bf16, f32> for (bf16, f32) {}

pub struct CmmaInstructionConfig {}

#[cube]
impl<I: Numeric, O: Numeric> MatmulInstruction<I, O> for CmmaInstruction16_16_16<I, O>
where
    (I, O): CmmaValid<I, O>,
{
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

#[cube]
impl<I: Numeric, O: Numeric> MatmulInstruction<I, O> for CmmaInstruction32_8_16<I, O>
where
    (I, O): CmmaValid<I, O>,
{
    type Config = CmmaInstructionConfig;
    type Lhs = cmma::Matrix<I>;
    type Rhs = cmma::Matrix<I>;
    type Out = cmma::Matrix<O>;
    const M: u32 = 32;
    const N: u32 = 8;
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

#[cube]
impl<I: Numeric, O: Numeric> MatmulInstruction<I, O> for CmmaInstruction8_32_16<I, O>
where
    (I, O): CmmaValid<I, O>,
{
    type Config = CmmaInstructionConfig;
    type Lhs = cmma::Matrix<I>;
    type Rhs = cmma::Matrix<I>;
    type Out = cmma::Matrix<O>;
    const M: u32 = 8;
    const N: u32 = 32;
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
