use crate::matmul::config::PlaneMapper;
use crate::matmul::matmul_tile::TileMatmul;
use crate::matmul::matmul_tile::TmmConfig;
use crate::matmul::matrix::Ident;
use crate::matmul::matrix::MatrixLayout;
use crate::matmul::Matmul;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

pub type PlaneMma16x16x16<I, O, T> = PlaneMma16x16<I, O, T, 16>;

pub struct PlaneMma16x16<I: Numeric, O: Numeric, T: TmmConfig, const K: u32> {
    _input: PhantomData<I>,
    _output: PhantomData<O>,
    _config: PhantomData<T>,
}

#[cube]
impl<I: Numeric, O: Numeric, T: TmmConfig, const K: u32> PlaneMapper for PlaneMma16x16<I, O, T, K> {
    fn plane_id() -> u32 {
        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }
}

#[cube]
impl<I: Numeric, O: Numeric, T: TmmConfig, const K: u32> TileMatmul<I, O, T>
    for PlaneMma16x16<I, O, T, K>
{
    const M: u32 = 16;
    const N: u32 = 16;
    const K: u32 = K;

    type Lhs = Array<I>;
    type Rhs = Array<I>;
    type Out = Array<O>;

    fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out) {
        // TODO config
        let plane_dim = 32;

        let k_jump = plane_dim / Self::K;
        let row_division = plane_dim / Self::M;
        let num_jumps = Self::K / k_jump;
        for k_iter in 0..num_jumps {
            let b_kp = rhs[k_iter];
            let compute_width = Self::N / row_division;
            let offset = Self::plane_unit() % row_division * compute_width;

            #[unroll]
            for k_ in 0..k_jump {
                let k = k_iter * k_jump + k_;
                let a_pk = lhs[k];

                #[unroll]
                for n_ in 0..compute_width {
                    let n = offset + n_;
                    let unit_to_read = n + k_ * Self::N;
                    let b_kn = subcube_broadcast::<I>(b_kp, unit_to_read);
                    out[n_] += O::cast_from(a_pk * b_kn);
                }
            }
        }
    }

    fn init_lhs(#[comptime] _config: T) -> Self::Lhs {
        Array::new(Self::K)
    }

    fn init_rhs(#[comptime] config: T) -> Self::Rhs {
        // 32/16 = 2
        let k_jump = config.plane_dim() / Self::K;
        // 16/2 = 8
        let num_jumps = Self::K / k_jump;
        Array::new(num_jumps)
    }

    fn fill_lhs(slice: &Slice<'_, Line<I>>, lhs: &mut Self::Lhs, #[comptime] config: T) {
        match comptime!(config.layout(Ident::Lhs)) {
            MatrixLayout::RowMajor => fill_parallel(
                slice,
                lhs.as_slice_mut(),
                Self::plane_unit(),
                Self::M,
                Self::K,
                config.line_size(Ident::Lhs),
                config.plane_dim(),
            ),
            MatrixLayout::ColMajor => fill_perpendicular(
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
            MatrixLayout::RowMajor => fill_perpendicular::<I>(
                slice,
                rhs.as_slice_mut(),
                Self::plane_unit(),
                Self::N,
                Self::K,
                config.line_size(Ident::Rhs),
                config.plane_dim(),
            ),
            MatrixLayout::ColMajor => fill_parallel(
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

    fn init_output() -> Self::Out {
        // TODO config
        let plane_dim = 32;

        // 16 * 16 / 32 = 8
        let len = Self::M * Self::N / plane_dim;
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

        // if UNIT_POS_X % 2 == 1 {
        let offset = unit % row_division * num_lines;
        for col_ in 0..num_lines {
            let col = col_;
            // let line = if comptime!(line_size == 1) {
            //     Line::cast_from(out[col])
            // } else {
            let line = {
                let mut line = Line::<C>::empty(line_size);
                for j in 0..line_size {
                    line[j] = C::cast_from(out[col * line_size + j]);
                }
                line
            };

            slice[row_offset + col + offset] = line;
        }
        // }
    }
}

#[cube]
fn fill_parallel<E: Numeric>(
    slice_from: &Slice<'_, Line<E>>,
    slice_to: &mut SliceMut<'_, E>,
    unit: u32,
    num_rows: u32,
    k: u32,
    #[comptime] line_size: u32,
    #[comptime] plane_dim: u32,
) {
    // 16/4 = 4
    let num_lines = k / line_size;
    // 32/16 = 2
    let row_division = plane_dim / num_rows;
    // 0..31/2 = 0..15
    let row = unit / row_division;

    // for col in 0..4
    for col in 0..num_lines {
        // 0..15 * 4 + 0..4 = 0....60 + 0..4 = 0..64
        let line = slice_from[row * num_lines + col];
        #[unroll]
        // for line_idx = 0..4
        for line_idx in 0..line_size {
            // 0..4 * 4 + 0..4 = 0....12 + 0..4 = 0..15
            slice_to[col * line_size + line_idx] = line[line_idx];
        }
    }
}

#[cube]
fn fill_perpendicular<E: Numeric>(
    slice_from: &Slice<'_, Line<E>>,
    slice_to: &mut SliceMut<'_, E>,
    unit: u32,                 // 0..31
    #[comptime] num_cols: u32, // 16
    #[comptime] k: u32,        // 16
    #[comptime] line_size: u32,
    #[comptime] plane_dim: u32,
) {
    // 0 or 1
    let k_row_alt = unit / num_cols;
    // 0..15
    let col = unit % num_cols;

    // 16/4 = 4
    let num_lines = num_cols / line_size;
    // 0..15/4 = 0..4
    let col_idx = col / line_size;
    // 0..15%4 = 0..4
    let line_idx = col % line_size;

    // 32/16=2
    let row_jump = plane_dim / num_cols;

    // 0..(16 / 2)=0..8
    #[unroll]
    for k_iter in 0..k / row_jump {
        // 2 * 0..8 + 0..1 = 0..15
        let k_row = row_jump * k_iter + k_row_alt;
        // 0..15 * 4 = 0....64
        let row_offset = k_row * num_lines;
        // 0....64 + 0..4 = 0..64
        let offset_with_col = row_offset + col_idx;
        // at line_idx=0..4 -> 256 elements
        // k_iter = 0..8, times 32 units -> 256 elements
        slice_to[k_iter] = slice_from[offset_with_col][line_idx];
    }
}

impl<I: Numeric, O: Numeric, T: TmmConfig, const K: u32> Matmul<I, O>
    for PlaneMma16x16<I, O, T, K>
{
    type Config = T;

    fn check_config(config: Self::Config) {
        assert!(config.plane_dim() == 32);
        assert!(K % 2 == 0 && K > 0)
    }
}
