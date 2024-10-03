use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::server::ComputeServer;
use cubecl_core::{cmma, prelude::*};
use half::{bf16, f16};

use crate::matmul::cmma_matmul::{BlockInfo, BlockInfos};
use crate::matmul::launch::matmul_instruction_launch;
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::{FixedShapeMatmul, Matmul, MatmulInstruction};

use super::implementation::*;

pub trait CmmaValid<I: Numeric, O: Numeric> {}
impl CmmaValid<f16, f16> for (f16, f16) {}
impl CmmaValid<f16, f32> for (f16, f32) {}
impl CmmaValid<bf16, f32> for (bf16, f32) {}

pub struct CmmaInstructionConfig {}

#[derive(CubeType)]
pub struct Fragment<T: Numeric> {
    pub matrix: cmma::Matrix<T>,
    pub stride: u32,
}

macro_rules! impl_matmul_instruction {
    ($name:ident, $m:expr, $n:expr, $k:expr) => {
        pub struct $name<I: Numeric, O: Numeric> {
            _input: PhantomData<I>,
            _output: PhantomData<O>,
        }

        impl<I: Numeric, O: Numeric> FixedShapeMatmul<I, O> for $name<I, O>
        where
            (I, O): CmmaValid<I, O>,
        {
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

        impl<I: Numeric, O: Numeric> Matmul<I, O> for $name<I, O>
        where
            (I, O): CmmaValid<I, O>,
        {
            fn cube_dim_resources() -> CubeDim {
                CubeDim::new(32, 1, 1)
            }

            fn cube_count_resources<S: ComputeServer>() -> CubeCount<S> {
                CubeCount::Static(1, 1, 1)
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
        impl<I: Numeric, O: Numeric> MatmulInstruction<I, O> for $name<I, O>
        where
            (I, O): CmmaValid<I, O>,
        {
            type Config = CmmaInstructionConfig;
            type Lhs = Fragment<I>;
            type Rhs = Fragment<I>;
            type Out = Fragment<O>;

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
                fill_lhs(slice, lhs);
            }

            fn fill_rhs<C: CubePrimitive>(slice: &Slice<'_, C>, rhs: &mut Self::Rhs) {
                fill_rhs(slice, rhs);
            }

            fn init_output() -> Self::Out {
                init_output(Self::M, Self::N, Self::K)
            }

            fn read_output<C: CubePrimitive>(out: &Self::Out, slice: &mut SliceMut<'_, C>) {
                read_output::<O, C>(out, slice);
            }
        }
    };
}

impl_matmul_instruction!(CmmaInstruction16_16_16, 16, 16, 16);
impl_matmul_instruction!(CmmaInstruction32_8_16, 32, 8, 16);
impl_matmul_instruction!(CmmaInstruction8_32_16, 8, 32, 16);
