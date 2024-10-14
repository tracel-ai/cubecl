use crate::matmul::launch::matmul_instruction_launch;
use crate::matmul::matmul_tile::MatmulInstruction;
use crate::matmul::matmul_tile::OwnedTile;
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::problem::{MatmulProblem, Requirements};
use crate::matmul::stage_info::{StageInfo, StageInfos};
use crate::matmul::{FixedShapeMatmul, Matmul};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

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
                cube_count: CubeCount,
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

            fn stage_infos() -> StageInfos {
                StageInfos {
                    lhs: StageInfo {
                        num_tiles_x: 1,
                        num_tiles_y: 1,
                        tile_size_x: $m,
                        tile_size_y: $k,
                    },
                    rhs: StageInfo {
                        num_tiles_x: 1,
                        num_tiles_y: 1,
                        tile_size_x: $k,
                        tile_size_y: $n,
                    },
                    out: StageInfo {
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
            type Lhs = OwnedTile<I>;
            type Rhs = OwnedTile<I>;
            type Out = OwnedTile<O>;

            fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out) {
                execute::<I, O>(lhs, rhs, out);
            }

            fn init_lhs(layout: MatrixLayout) -> Self::Lhs {
                OwnedTile::<I> {
                    handle: Array::<I>::new(Self::M * Self::K),
                    x: Self::M.runtime(),
                    y: Self::K.runtime(),
                    layout,
                }
            }

            fn init_rhs(layout: MatrixLayout) -> Self::Rhs {
                OwnedTile::<I> {
                    handle: Array::<I>::new(Self::K * Self::N),
                    x: Self::K.runtime(),
                    y: Self::N.runtime(),
                    layout,
                }
            }

            fn fill_lhs(slice: &Slice<'_, Line<I>>, lhs: &mut Self::Lhs) {
                asdf(slice, lhs);
                // for i in 0..Self::M * Self::K {
                //     lhs.handle[i] = Line::cast_from(slice[i]);
                // }
            }

            fn fill_rhs(slice: &Slice<'_, Line<I>>, rhs: &mut Self::Rhs) {
                asdf(slice, rhs)
                // for i in 0..Self::K * Self::N {
                //     rhs.handle[i] = Line::cast_from(slice[i]);
                // }
            }

            fn init_output() -> Self::Out {
                let mut out = OwnedTile::<O> {
                    handle: Array::<O>::new(Self::M * Self::N),
                    x: Self::M.runtime(),
                    y: Self::N.runtime(),
                    layout: MatrixLayout::RowMajor.runtime(),
                };

                for i in 0..Self::M * Self::N {
                    out.handle[i] = O::from_int(0);
                }

                out
            }

            fn read_output<C: Numeric>(out: &Self::Out, slice: &mut SliceMut<'_, Line<C>>) {
                let line_size = Line::size(&slice[0]);
                for i in 0..out.x * out.y / line_size {
                    let mut line = Line::<C>::empty(line_size);
                    for j in 0..line_size {
                        line[j] = C::cast_from(out.handle[i * line_size + j]);
                    }
                    slice[i] = line;
                }
            }
        }
    };
}

impl_matmul_instruction!(DummyUnitInstruction16_16_16, 16, 16, 16);
impl_matmul_instruction!(DummyUnitInstruction32_8_16, 32, 8, 16);
impl_matmul_instruction!(DummyUnitInstruction8_32_16, 8, 32, 16);

#[cube]
fn asdf<E: Numeric>(slice: &Slice<'_, Line<E>>, tile: &mut OwnedTile<E>) {
    // TODO is there better way
    let line_size = Line::size(&slice[0]);

    for i in 0..tile.x * tile.y / line_size {
        let line = slice[i];
        for j in 0..line_size {
            tile.handle[i * line_size + j] = line[j];
        }
    }
}

// #[cube]
// fn qwerty<E: Numeric>(tile: &mut OwnedTile<E>, slice: &SliceMut<'_, Line<E>>) {
//     // TODO is there better way
//     let line_size = Line::size(&slice[0]);

//     for i in 0..tile.x * tile.y / line_size {
//         let line = tile.handle[i];
//         for j in 0..line_size {
//             tile.handle[i * line_size + j] = Line::cast_from(line[j]);
//         }
//     }
// }

#[cube]
pub(crate) fn execute<I: Numeric, O: Numeric>(
    lhs: &OwnedTile<I>,
    rhs: &OwnedTile<I>,
    out: &mut OwnedTile<O>,
) {
    let m = lhs.x;
    let n = rhs.y;
    let k = rhs.x;

    for i in 0..m {
        for j in 0..n {
            for k_ in 0..k {
                let lhs_val = match comptime!(lhs.layout) {
                    MatrixLayout::RowMajor => lhs.handle[i * k + k_],
                    MatrixLayout::ColMajor => lhs.handle[k_ * m + i],
                };
                let rhs_val = match comptime!(rhs.layout) {
                    MatrixLayout::RowMajor => rhs.handle[k_ * n + j],
                    MatrixLayout::ColMajor => rhs.handle[j * k + k_],
                };

                out.handle[i * n + j] += O::cast_from(lhs_val * rhs_val);
            }
        }
    }
}
