use super::implementation::*;
use crate::matmul::matmul_tile::TmmConfig;
use crate::matmul::matmul_tile::TileMatmul;
use crate::matmul::matrix::MatrixLayout;
use crate::matmul::Matmul;
use cubecl_core as cubecl;
use cubecl_core::{cmma, prelude::*};
use half::{bf16, f16};
use std::marker::PhantomData;

pub trait CmmaValid<I: Numeric, O: Numeric> {}

impl CmmaValid<f16, f16> for (f16, f16) {}
impl CmmaValid<f16, f32> for (f16, f32) {}
impl CmmaValid<bf16, f32> for (bf16, f32) {}

#[derive(CubeType)]
pub struct Fragment<T: Numeric> {
    pub matrix: cmma::Matrix<T>,
    pub stride: u32,
}

macro_rules! impl_matmul_instruction {
    ($name:ident, $m:expr, $n:expr, $k:expr) => {
        pub struct $name<I: Numeric, O: Numeric, T: TmmConfig> {
            _input: PhantomData<I>,
            _output: PhantomData<O>,
            _config: PhantomData<T>,
        }

        #[cube]
        impl<I: Numeric, O: Numeric, T: TmmConfig> TileMatmul<I, O> for $name<I, O, T>
        where
            (I, O): CmmaValid<I, O>,
        {
            const M: u32 = $m;
            const N: u32 = $n;
            const K: u32 = $k;

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

            fn fill_lhs(slice: &Slice<'_, Line<I>>, lhs: &mut Self::Lhs) {
                fill_lhs(slice, lhs);
            }

            fn fill_rhs(slice: &Slice<'_, Line<I>>, rhs: &mut Self::Rhs) {
                fill_rhs(slice, rhs);
            }

            fn init_output() -> Self::Out {
                init_output(Self::M, Self::N, Self::K)
            }

            fn read_output<C: Numeric>(out: &Self::Out, slice: &mut SliceMut<'_, Line<C>>) {
                read_output::<O, C>(out, slice);
            }
        }

        impl<I: Numeric, O: Numeric, T: TmmConfig> Matmul<I, O> for $name<I, O, T>
        where
            (I, O): CmmaValid<I, O>,
        {
            type Config = T;
            
            // fn preconfigure() -> CmmaPreConfig {
            //     CmmaPreConfig {
            //         lhs_tile_size_x: $m,
            //         lhs_tile_size_y: $k,
            //         rhs_tile_size_x: $k,
            //         rhs_tile_size_y: $n,
            //         out_tile_size_x: $m,
            //         out_tile_size_y: $n,
            //         ..Default::default()
            //     }
            // }

            fn check_config(config: Self::Config) {
                let _ = comptime!(check_plane_dim(config.plane_dim()));
            }

            // unsafe fn launch_unchecked<R: Runtime>(
            //     client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
            //     cube_dim: CubeDim,
            //     cube_count: CubeCount,
            //     lhs: TensorArg<'_, R>,
            //     rhs: TensorArg<'_, R>,
            //     out: TensorArg<'_, R>,
            //     config: CmmaConfig,
            // ) {
            //     Self::check_config(config);
            //     matmul_instruction_launch::launch_unchecked::<Self, I, O, R>(
            //         &client,
            //         cube_count,
            //         cube_dim,
            //         lhs,
            //         rhs,
            //         out,
            //         config.layouts,
            //     );
            // }
        }
    };
}

impl_matmul_instruction!(CmmaInstruction16_16_16, 16, 16, 16);
impl_matmul_instruction!(CmmaInstruction32_8_16, 32, 8, 16);
impl_matmul_instruction!(CmmaInstruction8_32_16, 8, 32, 16);

fn check_plane_dim(actual_plane_dim: u32) {
    assert_eq!(32, actual_plane_dim, 
        "Error: Expected plane dimension to be 32, but found {}. Please ensure that cube dimension x is set correctly.",
        actual_plane_dim
    );
}
