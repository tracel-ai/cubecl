use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::{cmma, prelude::*};
use half::{bf16, f16};

use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::MatmulInstruction;

use super::implementation::*;

pub trait CmmaValid<I: Numeric, O: Numeric> {}
impl CmmaValid<f16, f16> for (f16, f16) {}
impl CmmaValid<f16, f32> for (f16, f32) {}
impl CmmaValid<bf16, f32> for (bf16, f32) {}

pub struct CmmaInstructionConfig {}

macro_rules! impl_matmul_instruction {
    ($name:ident, $m:expr, $n:expr, $k:expr) => {
        pub struct $name<I: Numeric, O: Numeric> {
            _input: PhantomData<I>,
            _output: PhantomData<O>,
        }

        #[cube]
        impl<I: Numeric, O: Numeric> MatmulInstruction<I, O> for $name<I, O>
        where
            (I, O): CmmaValid<I, O>,
        {
            type Config = CmmaInstructionConfig;
            type Lhs = cmma::Matrix<I>;
            type Rhs = cmma::Matrix<I>;
            type Out = cmma::Matrix<O>;
            const M: u32 = $m;
            const N: u32 = $n;
            const K: u32 = $k;

            fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out) {
                execute::<I, O>(lhs, rhs, out);
            }

            fn init_lhs(#[comptime] layout: MatrixLayout) -> Self::Lhs {
                init_lhs(layout, Self::M, Self::N, Self::K)
            }

            fn init_rhs(#[comptime] layout: MatrixLayout) -> Self::Rhs {
                init_rhs(layout, Self::M, Self::N, Self::K)
            }

            fn fill_lhs<C: CubePrimitive>(
                slice: &Slice<'_, C>,
                lhs: &mut Self::Lhs,
                #[comptime] layout: MatrixLayout,
            ) {
                fill_lhs(slice, lhs, Self::M, Self::K, layout);
            }

            fn fill_rhs<C: CubePrimitive>(
                slice: &Slice<'_, C>,
                rhs: &mut Self::Rhs,
                #[comptime] layout: MatrixLayout,
            ) {
                fill_rhs(slice, rhs, Self::K, Self::N, layout);
            }

            fn init_output() -> Self::Out {
                init_output(Self::M, Self::N, Self::K)
            }

            fn read_output<C: CubePrimitive>(out: &Self::Out, slice: &mut SliceMut<'_, C>) {
                read_output::<O, C>(out, slice, Self::N);
            }
        }
    };
}

impl_matmul_instruction!(CmmaInstruction16_16_16, 16, 16, 16);
impl_matmul_instruction!(CmmaInstruction32_8_16, 32, 8, 16);
impl_matmul_instruction!(CmmaInstruction8_32_16, 8, 32, 16);
