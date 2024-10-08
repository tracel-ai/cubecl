use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::block_info::{BlockInfo, BlockInfos};
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::MatmulInstruction;

use crate::matmul::problem::{MatmulProblem, Requirements};

use crate::matmul::launch::matmul_instruction_launch;
use crate::matmul::{FixedShapeMatmul, Matmul};

pub struct DummyUnitInstructionConfig {}

#[derive(CubeType)]
pub struct DummyMatrix<E: Numeric> {
    pub handle: Array<E>,
    pub shape: (u32, u32),
    pub is_col_major: bool,
}

macro_rules! impl_matmul_instruction {
    ($name:ident, $m:expr, $n:expr, $k:expr) => {
        /// All units in a plane do the same work in parallel
        ///
        /// Useful to mimic behaviour of CMMA when the instruction is unavailable
        /// Has lots of duplication and likely lots of bank conflicts
        pub struct $name<I: Numeric, O: Numeric> {
            _input: PhantomData<I>,
            _output: PhantomData<O>,
        }

        impl<I: Numeric, O: Numeric> FixedShapeMatmul<I, O> for $name<I, O> {
            const M: u32 = $m;
            const N: u32 = $n;
            const K: u32 = $k;

            unsafe fn launch_unchecked<R: Runtime>(
                client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
                cube_dim: CubeDim,
                cube_count: CubeCount<<R as Runtime>::Server>,
                lhs: ArrayArg<'_, R>,
                rhs: ArrayArg<'_, R>,
                out: ArrayArg<'_, R>,
                layouts: (MatrixLayout, MatrixLayout),
            ) {
                matmul_instruction_launch::launch_unchecked::<Self, I, O, R>(
                    &client, cube_count, cube_dim, lhs, rhs, out, layouts,
                );
            }
        }

        impl<I: Numeric, O: Numeric> Matmul<I, O> for $name<I, O> {
            fn can_process(problem: MatmulProblem) -> bool {
                problem.m == Self::M && problem.n == Self::N && problem.k == Self::K
            }

            fn requirements(_problem: MatmulProblem) -> Requirements {
                Requirements {
                    num_planes: 1,
                    num_cubes: 1,
                }
            }

            fn block_infos() -> BlockInfos {
                BlockInfos {
                    lhs: BlockInfo {
                        num_tiles_x: 1,
                        num_tiles_y: 1,
                        tile_size_x: $m,
                        tile_size_y: $k,
                    },
                    rhs: BlockInfo {
                        num_tiles_x: 1,
                        num_tiles_y: 1,
                        tile_size_x: $k,
                        tile_size_y: $n,
                    },
                    out: BlockInfo {
                        num_tiles_x: 1,
                        num_tiles_y: 1,
                        tile_size_x: $m,
                        tile_size_y: $n,
                    },
                }
            }
        }

        #[cube]
        impl<I: Numeric, O: Numeric> MatmulInstruction<I, O> for $name<I, O> {
            type Config = DummyUnitInstructionConfig;
            type Lhs = DummyMatrix<I>;
            type Rhs = DummyMatrix<I>;
            type Out = DummyMatrix<O>;

            fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out) {
                execute::<I, O>(lhs, rhs, out);
            }

            fn init_lhs(#[comptime] layout: MatrixLayout) -> Self::Lhs {
                DummyMatrix::<I> {
                    handle: Array::<I>::new(Self::M * Self::K),
                    shape: (Self::M.runtime(), Self::K.runtime()),
                    is_col_major: match layout {
                        MatrixLayout::RowMajor => false,
                        MatrixLayout::ColMajor => true,
                    },
                }
            }

            fn init_rhs(#[comptime] layout: MatrixLayout) -> Self::Rhs {
                DummyMatrix::<I> {
                    handle: Array::<I>::new(Self::K * Self::N),
                    shape: (Self::K.runtime(), Self::N.runtime()),
                    is_col_major: match layout {
                        MatrixLayout::RowMajor => false,
                        MatrixLayout::ColMajor => true,
                    },
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
                    is_col_major: false,
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
                let lhs_val = if lhs.is_col_major {
                    lhs.handle[k_ * m + i]
                } else {
                    lhs.handle[i * k + k_]
                };
                let rhs_val = if rhs.is_col_major {
                    rhs.handle[j * k + k_]
                } else {
                    rhs.handle[k_ * n + j]
                };

                out.handle[i * n + j] += O::cast_from(lhs_val * rhs_val);
            }
        }
    }
}
