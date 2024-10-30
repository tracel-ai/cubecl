use crate::matmul::matmul_modular::matrix::as_cmma_layout;
use crate::matmul::matmul_modular::matmul_tile::TmmConfig;
use crate::matmul::matmul_modular::matmul_tile::TileMatmul;
use crate::matmul::matmul_modular::matrix::MatrixLayout;
use crate::matmul::matmul_modular::matrix::Ident;
use crate::matmul::matmul_modular::Matmul;
use cubecl_core as cubecl;
use cubecl_core::{cmma, prelude::*};
use half::{bf16, f16};
use std::marker::PhantomData;

/// Implementations are pairs of element types that are allowed for CMMA
pub trait CmmaValid<I: Numeric, O: Numeric> {}

impl CmmaValid<f16, f16> for (f16, f16) {}
impl CmmaValid<f16, f32> for (f16, f32) {}
impl CmmaValid<bf16, f32> for (bf16, f32) {}

#[derive(CubeType)]
/// Wrapper over a CMMA matrix, containing the stride which implies the layout
pub struct Fragment<T: Numeric> {
    pub matrix: cmma::Matrix<T>,
    pub stride: u32,
}

/// CMMA instruction with m=16, n=16, k=16
pub struct CmmaInstruction16_16_16<I: Numeric, O: Numeric, T: TmmConfig> {
    _input: PhantomData<I>,
    _output: PhantomData<O>,
    _config: PhantomData<T>,
}

/// CMMA instruction with m=32, n=8, k=16
pub struct CmmaInstruction32_8_16<I: Numeric, O: Numeric, T: TmmConfig> {
    _input: PhantomData<I>,
    _output: PhantomData<O>,
    _config: PhantomData<T>,
}

/// CMMA instruction with m=8, n=32, k=16
pub struct CmmaInstruction8_32_16<I: Numeric, O: Numeric, T: TmmConfig> {
    _input: PhantomData<I>,
    _output: PhantomData<O>,
    _config: PhantomData<T>,
}

#[cube]
impl<I: Numeric, O: Numeric, T: TmmConfig> TileMatmul<I, O, T> for CmmaInstruction16_16_16<I, O, T>
where
    (I, O): CmmaValid<I, O>,
{
    const M: u32 = 16;
    const N: u32 = 16;
    const K: u32 = 16;

    type Lhs = Fragment<I>;
    type Rhs = Fragment<I>;
    type Out = Fragment<O>;

    fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out, #[comptime] _config: T) {
        execute::<I, O>(lhs, rhs, out);
    }

    fn init_lhs(#[comptime] config: T) -> Self::Lhs {
        init_lhs(config.layout(Ident::Lhs), Self::M, Self::N, Self::K)
    }

    fn init_rhs(#[comptime] config: T) -> Self::Rhs {
        init_rhs(config.layout(Ident::Rhs), Self::M, Self::N, Self::K)
    }

    fn fill_lhs(slice: &Slice<'_, Line<I>>, lhs: &mut Self::Lhs, #[comptime] _config: T) {
        fill_lhs(slice, lhs);
    }

    fn fill_rhs(slice: &Slice<'_, Line<I>>, rhs: &mut Self::Rhs, #[comptime] _config: T) {
        fill_rhs(slice, rhs);
    }

    fn init_output(#[comptime] _config: T) -> Self::Out {
        init_output(Self::M, Self::N, Self::K)
    }

    fn read_output<C: Numeric>(out: &Self::Out, slice: &mut SliceMut<'_, Line<C>>, #[comptime] _config: T) {
        read_output::<O, C>(out, slice);
    }
}

#[cube]
impl<I: Numeric, O: Numeric, T: TmmConfig> TileMatmul<I, O, T> for CmmaInstruction32_8_16<I, O, T>
where
    (I, O): CmmaValid<I, O>,
{
    const M: u32 = 32;
    const N: u32 = 8;
    const K: u32 = 16;

    type Lhs = Fragment<I>;
    type Rhs = Fragment<I>;
    type Out = Fragment<O>;

    fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out, #[comptime] _config: T) {
        execute::<I, O>(lhs, rhs, out);
    }

    fn init_lhs(#[comptime] config: T) -> Self::Lhs {
        init_lhs(config.layout(Ident::Lhs), Self::M, Self::N, Self::K)
    }

    fn init_rhs(#[comptime] config: T) -> Self::Rhs {
        init_rhs(config.layout(Ident::Rhs), Self::M, Self::N, Self::K)
    }

    fn fill_lhs(slice: &Slice<'_, Line<I>>, lhs: &mut Self::Lhs, #[comptime] _config: T) {
        fill_lhs(slice, lhs);
    }

    fn fill_rhs(slice: &Slice<'_, Line<I>>, rhs: &mut Self::Rhs, #[comptime] _config: T) {
        fill_rhs(slice, rhs);
    }

    fn init_output(#[comptime] _config: T) -> Self::Out {
        init_output(Self::M, Self::N, Self::K)
    }

    fn read_output<C: Numeric>(out: &Self::Out, slice: &mut SliceMut<'_, Line<C>>, #[comptime] _config: T) {
        read_output::<O, C>(out, slice);
    }
}

#[cube]
impl<I: Numeric, O: Numeric, T: TmmConfig> TileMatmul<I, O, T> for CmmaInstruction8_32_16<I, O, T>
where
    (I, O): CmmaValid<I, O>,
{
    const M: u32 = 8;
    const N: u32 = 32;
    const K: u32 = 16;

    type Lhs = Fragment<I>;
    type Rhs = Fragment<I>;
    type Out = Fragment<O>;

    fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out, #[comptime] _config: T) {
        execute::<I, O>(lhs, rhs, out);
    }

    fn init_lhs(#[comptime] config: T) -> Self::Lhs {
        init_lhs(config.layout(Ident::Lhs), Self::M, Self::N, Self::K)
    }

    fn init_rhs(#[comptime] config: T) -> Self::Rhs {
        init_rhs(config.layout(Ident::Rhs), Self::M, Self::N, Self::K)
    }

    fn fill_lhs(slice: &Slice<'_, Line<I>>, lhs: &mut Self::Lhs, #[comptime] _config: T) {
        fill_lhs(slice, lhs);
    }

    fn fill_rhs(slice: &Slice<'_, Line<I>>, rhs: &mut Self::Rhs, #[comptime] _config: T) {
        fill_rhs(slice, rhs);
    }

    fn init_output(#[comptime] _config: T) -> Self::Out {
        init_output(Self::M, Self::N, Self::K)
    }

    fn read_output<C: Numeric>(out: &Self::Out, slice: &mut SliceMut<'_, Line<C>>, #[comptime] _config: T) {
        read_output::<O, C>(out, slice);
    }
}


impl<I: Numeric, O: Numeric, T: TmmConfig> Matmul<I, O> for CmmaInstruction16_16_16<I, O, T>
where
    (I, O): CmmaValid<I, O>,
{
    type Config = T;
    
    fn check_config(config: Self::Config) {
        let _ = comptime!(check_plane_dim(config.plane_dim()));
    }
}

impl<I: Numeric, O: Numeric, T: TmmConfig> Matmul<I, O> for CmmaInstruction32_8_16<I, O, T>
where
    (I, O): CmmaValid<I, O>,
{
    type Config = T;
    
    fn check_config(config: Self::Config) {
        let _ = comptime!(check_plane_dim(config.plane_dim()));
    }
}

impl<I: Numeric, O: Numeric, T: TmmConfig> Matmul<I, O> for CmmaInstruction8_32_16<I, O, T>
where
    (I, O): CmmaValid<I, O>,
{
    type Config = T;
    
    fn check_config(config: Self::Config) {
        let _ = comptime!(check_plane_dim(config.plane_dim()));
    }
}


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
