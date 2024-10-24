use crate::matmul::config::PlaneMapper;
use crate::matmul::matmul_tile::TileMatmul;
use crate::matmul::matmul_tile::TmmConfig;
use crate::matmul::matrix::Ident;
use crate::matmul::matrix::MatrixLayout;
use crate::matmul::Matmul;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

pub type PlaneMma32x32x32<I, O, T> = PlaneMma32x32<I, O, T, 32>;

pub struct PlaneMma32x32<I: Numeric, O: Numeric, T: TmmConfig, const K: u32> {
    _input: PhantomData<I>,
    _output: PhantomData<O>,
    _config: PhantomData<T>,
}

#[cube]
impl<I: Numeric, O: Numeric, T: TmmConfig, const K: u32> PlaneMapper for PlaneMma32x32<I, O, T, K> {
    fn plane_id() -> u32 {
        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }
}

#[cube]
impl<I: Numeric, O: Numeric, T: TmmConfig, const K: u32> TileMatmul<I, O, T>
    for PlaneMma32x32<I, O, T, K>
{
    const M: u32 = 32;
    const N: u32 = 32;
    const K: u32 = K;

    type Lhs = Array<I>;
    type Rhs = Array<I>;
    type Out = Array<O>;

    fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out, #[comptime] _config: T) {
        for k in 0..Self::K {
            let a_pk = lhs[k];
            let b_kp = rhs[k];

            #[unroll]
            for n in 0..Self::N {
                let b_kn = subcube_broadcast::<I>(b_kp, n);
                out[n] += O::cast_from(a_pk * b_kn);
            }
        }
    }

    fn init_lhs(#[comptime] _config: T) -> Self::Lhs {
        Array::new(Self::K)
    }

    fn init_rhs(#[comptime] _config: T) -> Self::Rhs {
        Array::new(Self::K)
    }

    fn fill_lhs(slice: &Slice<'_, Line<I>>, lhs: &mut Self::Lhs, #[comptime] config: T) {
        match comptime!(config.layout(Ident::Lhs)) {
            MatrixLayout::RowMajor => fill_parallel(
                slice,
                lhs.as_slice_mut(),
                Self::plane_unit(),
                Self::K,
                config.line_size(Ident::Lhs),
            ),
            MatrixLayout::ColMajor => fill_perpendicular(
                slice,
                lhs.as_slice_mut(),
                Self::plane_unit(),
                Self::M,
                Self::K,
                config.line_size(Ident::Lhs),
            ),
        }
    }

    fn fill_rhs(slice: &Slice<'_, Line<I>>, rhs: &mut Self::Rhs, #[comptime] config: T) {
        match comptime!(config.layout(Ident::Rhs)) {
            MatrixLayout::RowMajor => fill_perpendicular::<I>(
                slice,
                rhs.as_slice_mut(),
                Self::plane_unit(),
                Self::N,
                Self::K,
                config.line_size(Ident::Rhs),
            ),
            MatrixLayout::ColMajor => fill_parallel(
                slice,
                rhs.as_slice_mut(),
                Self::plane_unit(),
                Self::K,
                config.line_size(Ident::Rhs),
            ),
        }
    }

    fn init_output(#[comptime] _config: T) -> Self::Out {
        let mut acc = Array::new(Self::N);
        for i in 0..Self::N {
            acc[i] = O::from_int(0);
        }
        acc
    }

    fn read_output<C: Numeric>(
        out: &Self::Out,
        slice: &mut SliceMut<'_, Line<C>>,
        #[comptime] config: T,
    ) {
        let line_size = config.line_size(Ident::Out);
        let num_lines = Self::N / line_size;

        let row = Self::plane_unit();
        let unit_offset = row * num_lines;

        for col in 0..num_lines {
            let line = if comptime!(line_size == 1) {
                Line::cast_from(out[col])
            } else {
                let mut line = Line::<C>::empty(line_size);
                for j in 0..line_size {
                    line[j] = C::cast_from(out[col * line_size + j]);
                }
                line
            };

            slice[unit_offset + col] = line;
        }
    }
}

#[cube]
fn fill_parallel<E: Numeric>(
    slice_from: &Slice<'_, Line<E>>,
    slice_to: &mut SliceMut<'_, E>,
    row: u32,
    num_cols: u32,
    #[comptime] line_size: u32,
) {
    let num_lines = num_cols / line_size;

    for col in 0..num_lines {
        let line = slice_from[row * num_lines + col];
        #[unroll]
        for line_idx in 0..line_size {
            slice_to[col * line_size + line_idx] = line[line_idx];
        }
    }
}

#[cube]
fn fill_perpendicular<E: Numeric>(
    slice_from: &Slice<'_, Line<E>>,
    slice_to: &mut SliceMut<'_, E>,
    col: u32,
    num_cols: u32,
    num_rows: u32,
    #[comptime] line_size: u32,
) {
    // TODO: this loads a line then keeps only one element
    // Should reinterpret as unlined

    let num_lines = num_cols / line_size;
    let col_idx = col / line_size;
    let line_idx = col % line_size;

    for row in 0..num_rows {
        slice_to[row] = slice_from[row * num_lines + col_idx][line_idx];
    }
}

impl<I: Numeric, O: Numeric, T: TmmConfig, const K: u32> Matmul<I, O>
    for PlaneMma32x32<I, O, T, K>
{
    type Config = T;

    fn check_config(config: Self::Config) {
        assert!(config.plane_dim() == 32);
    }
}
