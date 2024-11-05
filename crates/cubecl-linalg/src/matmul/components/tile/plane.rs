use crate::matmul::components::config::MatmulConfig;
use crate::matmul::components::tile::Config as TileConfig;
use crate::matmul::components::{config::PlaneMapper, tile, Ident, MatmulKernel, MatrixLayout};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

pub type PlaneMma16x16x16<I, O> = PlaneMma<I, O, 16, 16, 16>;
pub type PlaneMma16x16x8<I, O> = PlaneMma<I, O, 16, 16, 8>;
pub type PlaneMma16x16x32<I, O> = PlaneMma<I, O, 16, 16, 32>;
pub type PlaneMma32x8x16<I, O> = PlaneMma<I, O, 32, 8, 16>;
pub type PlaneMma8x32x16<I, O> = PlaneMma<I, O, 8, 32, 16>;
pub type PlaneMma32x32x32<I, O> = PlaneMma<I, O, 32, 32, 32>;

/// PlaneMMA instruction uses plane cooperation but does not rely on tensor cores
///
/// # Note
///
/// This is not yet fully optimized
///  - There are likely unrolling issues,
///  - When loading perpendicular to the lines, too much data is loaded from in comparison to what is used
///  - To fix an obscure bug one loop is done twice
pub struct PlaneMma<I: Numeric, O: Numeric, const M: u32, const N: u32, const K: u32> {
    _input: PhantomData<I>,
    _output: PhantomData<O>,
}

#[cube]
impl<I: Numeric, O: Numeric, const M: u32, const N: u32, const K: u32> PlaneMapper
    for PlaneMma<I, O, M, N, K>
{
    fn plane_id() -> u32 {
        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }
}

#[cube]
impl<I: Numeric, O: Numeric, const M: u32, const N: u32, const K: u32> tile::Matmul<I, O, Config>
    for PlaneMma<I, O, M, N, K>
{
    const M: u32 = M;
    const N: u32 = N;
    const K: u32 = K;

    type Lhs = Array<Line<I>>;
    type Rhs = Array<Line<I>>;
    type Out = Array<O>;

    fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out, #[comptime] config: Config) {
        let output_len = Self::M * Self::N / config.plane_dim();
        let num_lines_k = Self::K / config.k_line_size;

        let unit = Self::plane_unit();
        let b_index = unit % (Self::N / output_len);

        let value_0 = rhs[0];
        let value_1 = rhs[1];

        #[unroll]
        for o_iter in 0..output_len {
            let mut val = out[o_iter];

            #[unroll]
            for k_iter in 0..num_lines_k {
                let a = lhs[k_iter];
                let broadcast_index = o_iter * num_lines_k + k_iter;
                let b0 = subcube_broadcast(value_0, broadcast_index);
                // let b1 = subcube_broadcast(value_1, broadcast_index);
                // let b = select(b_index == 0, b0, b1);

                val += O::cast_from(Line::dot(a, b0)[0]);
            }

            out[o_iter] = val;
        }

        // let num_out_lines = output_len / config.out_line_size;

        // #[unroll]
        // for o_outer in 0..num_out_lines {
        //     let mut line = Line::empty(config.out_line_size);

        //     #[unroll]
        //     for o_inner in 0..config.out_line_size {
        //         let mut val = O::from_int(0);
        //         let o = o_outer * config.out_line_size + o_inner;

        //         #[unroll]
        //         for k_iter in 0..num_lines_k {
        //             let a = lhs[k_iter];
        //             let broadcast_index = o * num_lines_k + k_iter;
        //             let b0 = subcube_broadcast(value_0, broadcast_index);
        //             let b1 = subcube_broadcast(value_1, broadcast_index);
        //             let b = select(b_index == 0, b0, b1);

        //             val += O::cast_from(Line::dot(a, b)[0]);
        //         }

        //         line[o_inner] = val;
        //     }

        //     out[o_outer] += line;
        // }
    }

    fn init_lhs(#[comptime] config: Config) -> Self::Lhs {
        let line_size = config.k_line_size;
        Array::vectorized(Self::K / line_size, line_size)
    }

    fn init_rhs(#[comptime] config: Config) -> Self::Rhs {
        let line_size = config.k_line_size;
        Array::vectorized(
            Self::K * Self::N / (line_size * config.plane_dim()),
            line_size,
        )
    }

    fn fill_lhs(slice: &Slice<'_, Line<I>>, lhs: &mut Self::Lhs, #[comptime] config: Config) {
        match comptime!(config.layout(Ident::Lhs)) {
            MatrixLayout::RowMajor => fill_parallel_lhs(
                slice,
                lhs.as_slice_mut(),
                Self::plane_unit(),
                Self::M,
                Self::K,
                config,
            ),
            MatrixLayout::ColMajor => fill_perpendicular_lhs(
                slice,
                lhs.as_slice_mut(),
                Self::plane_unit(),
                Self::M,
                Self::K,
                config,
            ),
        }
    }

    fn fill_rhs(slice: &Slice<'_, Line<I>>, rhs: &mut Self::Rhs, #[comptime] config: Config) {
        match comptime!(config.layout(Ident::Rhs)) {
            MatrixLayout::RowMajor => fill_perpendicular_rhs(
                slice,
                rhs.as_slice_mut(),
                Self::plane_unit(),
                Self::N,
                Self::K,
                config,
            ),
            MatrixLayout::ColMajor => fill_parallel_rhs(
                slice,
                rhs.as_slice_mut(),
                Self::plane_unit(),
                Self::N,
                Self::K,
                config,
            ),
        }
    }

    fn init_output(#[comptime] config: Config) -> Self::Out {
        let len = Self::M * Self::N / (config.plane_dim());
        let mut acc = Array::new(len);
        for i in 0..len {
            acc[i] = O::from_int(0);
        }
        acc
    }

    fn read_output<C: Numeric>(
        out: &Self::Out,
        slice: &mut SliceMut<'_, Line<C>>,
        #[comptime] config: Config,
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
            #[unroll]
            for col in 0..num_lines {
                slice[col + offset] = Line::cast_from(out[col]);
            }
            // TODO: very obscure bug, fails on wgpu if we don't do the loop twice
            for col in 0..num_lines {
                slice[col + offset] = Line::cast_from(out[col]);
            }
        } else {
            #[unroll]
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
    slice_to: &mut SliceMut<'_, Line<E>>,
    unit: u32,
    #[comptime] m: u32,
    #[comptime] k: u32,
    #[comptime] config: Config,
) {
    let lhs_line_size = config.line_size(Ident::Lhs);
    let plane_dim = config.plane_dim();

    let num_lines_to_fill = k / config.k_line_size;
    let row = unit * m / plane_dim;

    if comptime!(lhs_line_size == config.k_line_size) {
        #[unroll]
        for col in 0..num_lines_to_fill {
            slice_to[col] = slice_from[row * num_lines_to_fill + col];
        }
    } else {
        let _ = comptime!(unsupported_line_size());
    }
}

#[cube]
fn fill_perpendicular_lhs<E: Numeric>(
    slice_from: &Slice<'_, Line<E>>,
    slice_to: &mut SliceMut<'_, Line<E>>,
    unit: u32,
    #[comptime] m: u32,
    #[comptime] k: u32,
    #[comptime] config: Config,
) {
    let lhs_line_size = config.line_size(Ident::Lhs);
    let plane_dim = config.plane_dim();

    let num_lines_in_k = k / config.k_line_size;
    let col_stride = m / config.lhs_line_size;

    let row = unit * m / plane_dim;
    let lhs_row_idx = row / lhs_line_size;
    let lhs_line_idx = row % lhs_line_size;

    #[unroll]
    for col_outer in 0..num_lines_in_k {
        let mut line_to = Line::empty(config.k_line_size);

        #[unroll]
        for col_inner in 0..config.k_line_size {
            let col = col_outer * config.k_line_size + col_inner;
            let line_from = slice_from[lhs_row_idx + col * col_stride];
            line_to[col_inner] = line_from[lhs_line_idx];
        }

        slice_to[col_outer] = line_to;
    }
}

#[cube]
fn fill_perpendicular_rhs<E: Numeric>(
    slice_from: &Slice<'_, Line<E>>,
    slice_to: &mut SliceMut<'_, Line<E>>,
    unit: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
    #[comptime] config: Config,
) {
    let rhs_line_size = config.line_size(Ident::Rhs);
    let plane_dim = config.plane_dim();

    let num_lines_in_k = k / config.k_line_size;
    let row_stride_rhs = n / config.rhs_line_size;
    let col_jump = plane_dim / num_lines_in_k;
    let num_jumps = n / col_jump;

    let first_row = (unit % num_lines_in_k) * config.k_line_size;
    let col_unit_offset = unit / num_lines_in_k;

    #[unroll]
    for iter in 0..num_jumps {
        let mut line_to = Line::empty(config.k_line_size);

        let col = iter * col_jump + col_unit_offset;
        let rhs_col_idx = col / rhs_line_size;
        let rhs_line_idx = col % rhs_line_size;

        #[unroll]
        for row_inner in 0..config.k_line_size {
            let row = first_row + row_inner;
            let line_from = slice_from[row * row_stride_rhs + rhs_col_idx];
            line_to[row_inner] = line_from[rhs_line_idx];
        }

        slice_to[iter] = line_to;
    }
}

#[cube]
fn fill_parallel_rhs<E: Numeric>(
    slice_from: &Slice<'_, Line<E>>,
    slice_to: &mut SliceMut<'_, Line<E>>,
    unit: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
    #[comptime] config: Config,
) {
    let rhs_line_size = config.line_size(Ident::Rhs);
    let plane_dim = config.plane_dim();

    if comptime!(rhs_line_size == config.k_line_size) {
        let num_lines_in_k = k / config.k_line_size;

        let col_unit_offset = unit / num_lines_in_k;
        let col_jump = plane_dim / num_lines_in_k;
        let row = unit % num_lines_in_k;
        let num_jumps = n / col_jump;

        #[unroll]
        for col_iter in 0..num_jumps {
            let col = col_iter * col_jump + col_unit_offset;
            let line = slice_from[col * num_lines_in_k + row];
            slice_to[col_iter] = line;
        }
    } else {
        let _ = comptime!(unsupported_line_size());
    }
}

impl<I: Numeric, O: Numeric, const M: u32, const N: u32, const K: u32> MatmulKernel<I, O>
    for PlaneMma<I, O, M, N, K>
{
    type Config = Config;

    fn check_config(config: Self::Config) {
        let plane_dim = config.plane_dim();
        assert!(M * N % plane_dim == 0);
        assert!(K * N % plane_dim == 0);
    }
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for PlaneMma instruction
pub struct Config {
    plane_dim: u32,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    lhs_line_size: u32,
    rhs_line_size: u32,
    out_line_size: u32,
    k_line_size: u32,
}

impl tile::Config for Config {
    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }

    fn layout(&self, ident: Ident) -> MatrixLayout {
        match ident {
            Ident::Lhs => self.lhs_layout,
            Ident::Rhs => self.rhs_layout,
            Ident::Out => MatrixLayout::RowMajor,
        }
    }

    fn line_size(&self, ident: Ident) -> u32 {
        match ident {
            Ident::Lhs => self.lhs_line_size,
            Ident::Rhs => self.rhs_line_size,
            Ident::Out => self.out_line_size,
        }
    }
}

fn unsupported_line_size() {
    panic!("Different line sizes not supported yet")
}

impl MatmulConfig for Config {}

impl Config {
    pub fn new(
        plane_dim: u32,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        lhs_line_size: u32,
        rhs_line_size: u32,
        out_line_size: u32,
    ) -> Self {
        Self {
            plane_dim,
            lhs_layout,
            rhs_layout,
            lhs_line_size,
            rhs_line_size,
            out_line_size,
            // TODO make flexible
            k_line_size: 4,
        }
    }
}
