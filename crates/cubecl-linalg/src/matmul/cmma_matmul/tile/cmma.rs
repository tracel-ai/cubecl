use crate::matmul::matrix::as_cmma_layout;
use crate::matmul::matmul_tile::TmmConfig;
use crate::matmul::matmul_tile::TileMatmul;
use crate::matmul::matrix::MatrixLayout;
use crate::matmul::matrix::Ident;
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
        impl<I: Numeric, O: Numeric, T: TmmConfig> TileMatmul<I, O, T> for $name<I, O, T>
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

            fn init_lhs(#[comptime] config: T) -> Self::Lhs {
                init_lhs(config.layout(Ident::Lhs), Self::M, Self::N, Self::K)
            }

            fn init_rhs(#[comptime] config: T) -> Self::Rhs {
                init_rhs(config.layout(Ident::Rhs), Self::M, Self::N, Self::K)
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
            
            fn check_config(config: Self::Config) {
                let _ = comptime!(check_plane_dim(config.plane_dim()));
            }
        }
    };
}

impl_matmul_instruction!(CmmaInstruction16_16_16, 16, 16, 16);
impl_matmul_instruction!(CmmaInstruction32_8_16, 32, 8, 16);
impl_matmul_instruction!(CmmaInstruction8_32_16, 8, 32, 16);


#[cube]
pub(crate) fn execute<I: Numeric, O: Numeric>(
    lhs: &Fragment<I>,
    rhs: &Fragment<I>,
    out: &mut Fragment<O>,
) {
    cmma::execute::<I, I, O, O>(&lhs.matrix, &rhs.matrix, &out.matrix, &out.matrix);
}

#[cube]
pub(crate) fn init_lhs<I: Numeric>(
    #[comptime] layout: MatrixLayout,
    m: u32,
    n: u32,
    k: u32,
) -> Fragment<I> {
    unsafe {
        Fragment::<I> {
            matrix: cmma::Matrix::<I>::uninitialized(
                cmma::MatrixIdent::A,
                m,
                n,
                k,
                as_cmma_layout(layout),
            ),
            stride: match layout {
                MatrixLayout::RowMajor => k,
                MatrixLayout::ColMajor => m,
            },
        }
    }
}

#[cube]
pub(crate) fn init_rhs<I: Numeric>(
    #[comptime] layout: MatrixLayout,
    m: u32,
    n: u32,
    k: u32,
) -> Fragment<I> {
    unsafe {
        Fragment::<I> {
            matrix: cmma::Matrix::<I>::uninitialized(
                cmma::MatrixIdent::B,
                m,
                n,
                k,
                as_cmma_layout(layout),
            ),
            stride: match layout {
                MatrixLayout::RowMajor => n,
                MatrixLayout::ColMajor => k,
            },
        }
    }
}

#[cube]
pub(crate) fn fill_lhs<C: CubePrimitive, I: Numeric>(slice: &Slice<'_, C>, lhs: &mut Fragment<I>) {
    cmma::load(&lhs.matrix, slice, lhs.stride);
}

#[cube]
pub(crate) fn fill_rhs<C: CubePrimitive, I: Numeric>(slice: &Slice<'_, C>, rhs: &mut Fragment<I>) {
    cmma::load(&rhs.matrix, slice, rhs.stride);
}

#[cube]
pub(crate) fn init_output<O: Numeric>(m: u32, n: u32, k: u32) -> Fragment<O> {
    unsafe {
        let matrix = cmma::Matrix::<O>::uninitialized(
            cmma::MatrixIdent::Accumulator,
            m,
            n,
            k,
            cmma::MatrixLayout::Undefined,
        );

        cmma::fill(&matrix, O::from_int(0));

        Fragment::<O> { matrix, stride: n }
    }
}

#[cube]
pub(crate) fn read_output<O: Numeric, C: Numeric>(
    out: &Fragment<O>,
    slice: &mut SliceMut<'_, Line<C>>,
) {
    cmma::store(slice, &out.matrix, out.stride, cmma::MatrixLayout::RowMajor);
}

fn check_plane_dim(actual_plane_dim: u32) {
    assert_eq!(32, actual_plane_dim, 
        "Error: Expected plane dimension to be 32, but found {}. Please ensure that cube dimension x is set correctly.",
        actual_plane_dim
    );
}
