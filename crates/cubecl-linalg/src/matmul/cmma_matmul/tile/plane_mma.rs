use crate::matmul::config::PlaneMapper;
use crate::matmul::matmul_tile::TileMatmul;
use crate::matmul::matmul_tile::TmmConfig;
use crate::matmul::matrix::Ident;
use crate::matmul::matrix::MatrixLayout;
use crate::matmul::Matmul;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

pub type PlaneMma16x16x16<I, O, T> = PlaneMma<I, O, T, 16, 16, 16>;
pub type PlaneMma16x16x8<I, O, T> = PlaneMma<I, O, T, 16, 16, 8>;
pub type PlaneMma16x16x32<I, O, T> = PlaneMma<I, O, T, 16, 16, 32>;
pub type PlaneMma32x8x16<I, O, T> = PlaneMma<I, O, T, 32, 8, 16>;
pub type PlaneMma8x32x16<I, O, T> = PlaneMma<I, O, T, 8, 32, 16>;
pub type PlaneMma32x32x32<I, O, T> = PlaneMma<I, O, T, 32, 32, 32>;

pub struct PlaneMma<I: Numeric, O: Numeric, T: TmmConfig, const M: u32, const N: u32, const K: u32>
{
    _input: PhantomData<I>,
    _output: PhantomData<O>,
    _config: PhantomData<T>,
}

#[cube]
impl<I: Numeric, O: Numeric, T: TmmConfig, const M: u32, const N: u32, const K: u32> PlaneMapper
    for PlaneMma<I, O, T, M, N, K>
{
    fn plane_id() -> u32 {
        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }
}

#[cube]
impl<I: Numeric, O: Numeric, T: TmmConfig, const M: u32, const N: u32, const K: u32>
    TileMatmul<I, O, T> for PlaneMma<I, O, T, M, N, K>
{
    const M: u32 = M;
    const N: u32 = N;
    const K: u32 = K;

    type Lhs = Array<I>;
    type Rhs = Array<I>;
    type Out = Array<O>;

    fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out, #[comptime] config: T) {
        let k_jump = config.plane_dim() / Self::N;
        let row_division = config.plane_dim() / Self::M;

        let num_jumps = Self::K / k_jump;
        let compute_width = Self::N / row_division;

        let unit_offset = Self::plane_unit() % row_division * compute_width;

        #[unroll]
        for k_outer in 0..num_jumps {
            let b_kp = rhs[k_outer];

            #[unroll]
            for k_inner in 0..k_jump {
                let a_pk = lhs[k_outer * k_jump + k_inner];

                for n_iter in 0..compute_width {
                    let unit_to_read = k_inner * Self::N + n_iter + unit_offset;
                    let b_kn = subcube_broadcast::<I>(b_kp, unit_to_read);
                    out[n_iter] += O::cast_from(a_pk * b_kn);
                }
            }
        }
    }

    fn init_lhs(#[comptime] _config: T) -> Self::Lhs {
        Array::new(Self::K)
    }

    fn init_rhs(#[comptime] config: T) -> Self::Rhs {
        Array::new(Self::K * Self::N / config.plane_dim())
    }

    fn fill_lhs(slice: &Slice<'_, Line<I>>, lhs: &mut Self::Lhs, #[comptime] config: T) {
        match comptime!(config.layout(Ident::Lhs)) {
            MatrixLayout::RowMajor => fill_parallel_lhs(
                slice,
                lhs.as_slice_mut(),
                Self::plane_unit(),
                Self::M,
                Self::K,
                config.line_size(Ident::Lhs),
                config.plane_dim(),
            ),
            MatrixLayout::ColMajor => fill_perpendicular_lhs(
                slice,
                lhs.as_slice_mut(),
                Self::plane_unit(),
                Self::M,
                Self::K,
                config.line_size(Ident::Lhs),
                config.plane_dim(),
            ),
        }
    }

    fn fill_rhs(slice: &Slice<'_, Line<I>>, rhs: &mut Self::Rhs, #[comptime] config: T) {
        match comptime!(config.layout(Ident::Rhs)) {
            MatrixLayout::RowMajor => fill_perpendicular_rhs(
                slice,
                rhs.as_slice_mut(),
                Self::plane_unit(),
                Self::N,
                Self::K,
                config.line_size(Ident::Rhs),
                config.plane_dim(),
            ),
            MatrixLayout::ColMajor => fill_parallel_rhs(
                slice,
                rhs.as_slice_mut(),
                Self::plane_unit(),
                Self::N,
                Self::K,
                config.line_size(Ident::Rhs),
                config.plane_dim(),
            ),
        }
    }

    fn init_output(#[comptime] config: T) -> Self::Out {
        let len = Self::M * Self::N / config.plane_dim();
        let mut acc = Array::new(len);
        for i in 0..len {
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
        let plane_dim = config.plane_dim();

        let row_division = plane_dim / Self::M;
        let compute_width = Self::N / row_division;
        let num_lines = compute_width / line_size;

        let unit = Self::plane_unit();

        let row = unit / row_division;
        let row_offset = row * Self::N / line_size;

        let offset = row_offset + unit % row_division * num_lines;

        if comptime!(line_size == 1) {
            for col in 0..num_lines {
                slice[col + offset] = Line::cast_from(out[col]);
            }
            // TODO: very obscure bug, fails on wgpu if we don't do the loop twice
            for col in 0..num_lines {
                slice[col + offset] = Line::cast_from(out[col]);
            }
        } else {
            for col in 0..num_lines {
                let mut line = Line::<C>::empty(line_size);
                for j in 0..line_size {
                    line[j] = C::cast_from(out[col * line_size + j]);
                }
                slice[col + offset] = line;
            }
        }
    }
}

#[cube]
fn fill_parallel_lhs<E: Numeric>(
    slice_from: &Slice<'_, Line<E>>,
    slice_to: &mut SliceMut<'_, E>,
    unit: u32,
    #[comptime] m: u32,
    #[comptime] k: u32,
    #[comptime] line_size: u32,
    #[comptime] plane_dim: u32,
) {
    let num_lines = k / line_size;
    let row_division = plane_dim / m;
    let row = unit / row_division;

    #[unroll]
    for col in 0..num_lines {
        let line = slice_from[row * num_lines + col];
        if comptime!(line_size == 1) {
            slice_to[col * line_size] = E::cast_from(line);
        } else {
            #[unroll]
            for line_idx in 0..line_size {
                slice_to[col * line_size + line_idx] = line[line_idx];
            }
        }
    }
}

#[cube]
fn fill_perpendicular_lhs<E: Numeric>(
    slice_from: &Slice<'_, Line<E>>,
    slice_to: &mut SliceMut<'_, E>,
    unit: u32,
    #[comptime] m: u32,
    #[comptime] k: u32,
    #[comptime] line_size: u32,
    #[comptime] plane_dim: u32,
) {
    let num_lines = m / line_size;
    let row_division = plane_dim / m;
    let row = unit / row_division;

    let row_idx = row / line_size;
    let line_idx = row % line_size;

    #[unroll]
    for col in 0..k {
        slice_to[col] = if comptime!(line_size == 1) {
            E::cast_from(slice_from[row_idx + col * num_lines])
        } else {
            slice_from[row_idx + col * num_lines][line_idx]
        };
    }
}

#[cube]
fn fill_perpendicular_rhs<E: Numeric>(
    slice_from: &Slice<'_, Line<E>>,
    slice_to: &mut SliceMut<'_, E>,
    unit: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
    #[comptime] line_size: u32,
    #[comptime] plane_dim: u32,
) {
    // TODO: this loads a line then keeps only one element
    // Should reinterpret as unlined

    let k_row_alt = unit / n;
    let col = unit % n;

    let num_lines = n / line_size;
    let col_idx = col / line_size;
    let line_idx = col % line_size;

    let row_jump = plane_dim / n;

    #[unroll]
    for k_iter in 0..k / row_jump {
        let k_row = row_jump * k_iter + k_row_alt;
        let row_offset = k_row * num_lines;
        let offset_with_col = row_offset + col_idx;
        slice_to[k_iter] = if comptime!(line_size == 1) {
            E::cast_from(slice_from[offset_with_col])
        } else {
            slice_from[offset_with_col][line_idx]
        };
    }
}

#[cube]
fn fill_parallel_rhs<E: Numeric>(
    slice_from: &Slice<'_, Line<E>>,
    slice_to: &mut SliceMut<'_, E>,
    unit: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
    #[comptime] line_size: u32,
    #[comptime] plane_dim: u32,
) {
    let k_row_alt = unit / n;
    let col = unit % n;
    let row_jump = plane_dim / n;

    #[unroll]
    for k_iter in 0..k / row_jump {
        let row = row_jump * k_iter + k_row_alt;
        let row_index = row / line_size;
        let line_idx = row % line_size;
        let col_offset = col * k / line_size;
        let offset = row_index + col_offset;
        slice_to[k_iter] = if comptime!(line_size == 1) {
            E::cast_from(slice_from[offset])
        } else {
            slice_from[offset][line_idx]
        }
    }
}

impl<I: Numeric, O: Numeric, T: TmmConfig, const M: u32, const N: u32, const K: u32> Matmul<I, O>
    for PlaneMma<I, O, T, M, N, K>
{
    type Config = T;

    fn check_config(config: Self::Config) {
        let plane_dim = config.plane_dim();
        assert!(M * N % plane_dim == 0);
        assert!(K * N % plane_dim == 0);
    }
}
