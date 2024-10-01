use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::MatmulInstruction;

pub struct DummyUnitInstructionConfig {}

macro_rules! impl_matmul_instruction {
    ($name:ident, $m:expr, $n:expr, $k:expr) => {
        /// All units in a plane do the same work in parallel
        ///
        /// Useful to mimic behaviour of CMMA when the instruction is unavailable
        /// Likely has lots of bank conflicts
        pub struct $name<I: Numeric, O: Numeric> {
            _input: PhantomData<I>,
            _output: PhantomData<O>,
        }

        #[cube]
        impl<I: Numeric, O: Numeric> MatmulInstruction<I, O> for $name<I, O> {
            type Config = DummyUnitInstructionConfig;
            type Lhs = DummyMatrix<I>;
            type Rhs = DummyMatrix<I>;
            type Out = DummyMatrix<O>;
            const M: u32 = $m;
            const N: u32 = $n;
            const K: u32 = $k;

            fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out) {
                execute::<I, O>(lhs, rhs, out);
            }

            fn init_lhs(#[comptime] _layout: MatrixLayout) -> Self::Lhs {
                DummyMatrix::<I> {
                    handle: Array::<I>::new(Self::M * Self::K),
                    shape: (Self::M.runtime(), Self::K.runtime()),
                }
            }

            fn init_rhs(#[comptime] _layout: MatrixLayout) -> Self::Rhs {
                DummyMatrix::<I> {
                    handle: Array::<I>::new(Self::K * Self::N),
                    shape: (Self::K.runtime(), Self::N.runtime()),
                }
            }

            fn fill_lhs<C: CubePrimitive>(slice: &Slice<'_, C>, lhs: &mut Self::Lhs) {
                for i in 0..Self::M * Self::K {
                    lhs.handle[i] = I::cast_from(slice[i]);
                }
            }

            fn fill_rhs<C: CubePrimitive>(slice: &Slice<'_, C>, rhs: &mut Self::Rhs) {
                for i in 0..Self::K * Self::N {
                    rhs.handle[i] = I::cast_from(slice[i]);
                }
            }

            fn init_output() -> Self::Out {
                let mut out = DummyMatrix::<O> {
                    handle: Array::<O>::new(Self::M * Self::N),
                    shape: (Self::M.runtime(), Self::N.runtime()),
                };

                for i in 0..Self::M * Self::N {
                    out.handle[i] = O::from_int(0);
                }

                out
            }

            fn read_output<C: CubePrimitive>(out: &Self::Out, slice: &mut SliceMut<'_, C>) {
                for i in 0..256 {
                    slice[i] = C::cast_from(out.handle[i]);
                }
            }
        }
    };
}

impl_matmul_instruction!(DummyUnitInstruction16_16_16, 16, 16, 16);
impl_matmul_instruction!(DummyUnitInstruction32_8_16, 32, 8, 16);
impl_matmul_instruction!(DummyUnitInstruction8_32_16, 8, 32, 16);

#[derive(CubeType)]
pub struct DummyMatrix<E: Numeric> {
    pub handle: Array<E>,
    pub shape: (u32, u32),
}

#[cube]
pub(crate) fn execute<I: Numeric, O: Numeric>(
    lhs: &DummyMatrix<I>,
    rhs: &DummyMatrix<I>,
    out: &mut DummyMatrix<O>,
) {
    let m = lhs.shape.0;
    let n = rhs.shape.1;
    let k = rhs.shape.0;

    for i in 0..m {
        for j in 0..n {
            for k_ in 0..k {
                out.handle[i * n + j] +=
                    O::cast_from(lhs.handle[i * k + k_] * rhs.handle[k_ * n + j]);
            }
        }
    }
}
