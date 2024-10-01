use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::MatmulInstruction;

use super::implementation::*;

pub struct DummyUnitInstructionConfig {}

// macro_rules! impl_matmul_instruction {
//     ($name:ident, $m:expr, $n:expr, $k:expr) => {
//         pub struct $name<'a, I: Numeric, O: Numeric> {
//             _input: PhantomData<I>,
//             _output: PhantomData<O>,
//         }

//         #[cube]
//         impl<'a, I: Numeric, O: Numeric> MatmulInstruction<I, O> for $name<'a, I, O>
//         where
//             (I, O): CmmaValid<I, O>,
//         {
//             type Config = CmmaInstructionConfig;
//             type Lhs = Slice<'a, I>;
//             type Rhs = Slice<'a, I>;
//             type Out = Slice<'a, O>;
//             const M: u32 = $m;
//             const N: u32 = $n;
//             const K: u32 = $k;

//             fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out) {
//                 execute::<I, O>(lhs, rhs, out);
//             }

//             fn init_lhs(#[comptime] layout: MatrixLayout) -> Self::Lhs {
//                 init_lhs(layout, Self::M, Self::N, Self::K)
//             }

//             fn init_rhs(#[comptime] layout: MatrixLayout) -> Self::Rhs {
//                 init_rhs(layout, Self::M, Self::N, Self::K)
//             }

//             fn fill_lhs<C: CubePrimitive>(slice: &Slice<'_, C>, lhs: &mut Self::Lhs) {
//                 fill_lhs(slice, lhs, Self::K);
//             }

//             fn fill_rhs<C: CubePrimitive>(slice: &Slice<'_, C>, rhs: &mut Self::Rhs) {
//                 fill_rhs(slice, rhs, Self::N);
//             }

//             fn init_output() -> Self::Out {
//                 init_output(Self::M, Self::N, Self::K)
//             }

//             fn read_output<C: CubePrimitive>(out: &Self::Out, slice: &mut SliceMut<'_, C>) {
//                 read_output::<O, C>(out, slice, Self::N);
//             }
//         }
//     };
// }

// impl_matmul_instruction!(PlaneInstruction16_16_16, 16, 16, 16);
// impl_matmul_instruction!(PlaneInstruction32_8_16, 32, 8, 16);
// impl_matmul_instruction!(PlaneInstruction8_32_16, 8, 32, 16);

/// All units in a plane do the same work in parallel
///
/// Useful to mimic behaviour of CMMA when the instruction is unavailable
/// Likely has lots of bank conflicts
pub struct DummyUnitInstruction<I: Numeric, O: Numeric> {
    _input: PhantomData<I>,
    _output: PhantomData<O>,
}

#[derive(CubeType)]
pub struct DummyMatrix<E: Numeric> {
    pub handle: Array<E>,
    pub shape: (u32, u32),
}

#[cube]
impl<I: Numeric, O: Numeric> MatmulInstruction<I, O> for DummyUnitInstruction<I, O> {
    type Config = DummyUnitInstructionConfig;
    type Lhs = DummyMatrix<I>;
    type Rhs = DummyMatrix<I>;
    type Out = DummyMatrix<O>;
    const M: u32 = 16;
    const N: u32 = 16;
    const K: u32 = 16;

    fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out) {
        execute::<I, O>(lhs, rhs, out);
    }

    fn init_lhs(#[comptime] layout: MatrixLayout) -> Self::Lhs {
        init_lhs(layout, Self::M, Self::N, Self::K)
    }

    fn init_rhs(#[comptime] layout: MatrixLayout) -> Self::Rhs {
        init_rhs(layout, Self::M, Self::N, Self::K)
    }

    fn fill_lhs<C: CubePrimitive>(slice: &Slice<'_, C>, lhs: &mut Self::Lhs) {
        fill_lhs(slice, lhs, Self::K);
    }

    fn fill_rhs<C: CubePrimitive>(slice: &Slice<'_, C>, rhs: &mut Self::Rhs) {
        fill_rhs(slice, rhs, Self::N);
    }

    fn init_output() -> Self::Out {
        init_output(Self::M, Self::N, Self::K)
    }

    fn read_output<C: CubePrimitive>(out: &Self::Out, slice: &mut SliceMut<'_, C>) {
        read_output::<O, C>(out, slice, Self::N);
    }
}
