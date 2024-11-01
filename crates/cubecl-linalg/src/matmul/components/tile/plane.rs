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

    type Lhs = Array<I>;
    type Rhs = Array<I>;
    type Out = Array<O>;

    fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out, #[comptime] config: Config) {
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

    fn init_lhs(#[comptime] config: Config) -> Self::Lhs {
        let line_size = config.plane_mma_line_size;
        Array::vectorized(Self::K / line_size, line_size)
    }

    fn init_rhs(#[comptime] config: Config) -> Self::Rhs {
        let line_size = config.plane_mma_line_size;
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
    #[comptime] config: Config,
) {
    let lhs_line_size = config.line_size(Ident::Lhs);
    let plane_dim = config.plane_dim();

    let num_lines = k / lhs_line_size;
    let row = unit * m / plane_dim;

    #[unroll]
    for col in 0..num_lines {
        let line = slice_from[row * num_lines + col];
        if comptime!(lhs_line_size == 1) {
            slice_to[col * lhs_line_size] = E::cast_from(line);
        } else {
            #[unroll]
            for line_idx in 0..lhs_line_size {
                slice_to[col * lhs_line_size + line_idx] = line[line_idx];
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
    #[comptime] config: Config,
) {
    let lhs_line_size = config.line_size(Ident::Lhs);
    let plane_dim = config.plane_dim();

    let num_lines = m / lhs_line_size;
    let row = unit * m / plane_dim;

    let row_idx = row / lhs_line_size;
    let line_idx = row % lhs_line_size;

    #[unroll]
    for col in 0..k {
        let line = slice_from[row_idx + col * num_lines];
        slice_to[col] = if comptime!(lhs_line_size == 1) {
            E::cast_from(line)
        } else {
            line[line_idx]
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
    #[comptime] config: Config,
) {
    let rhs_line_size = config.line_size(Ident::Rhs);
    let plane_dim = config.plane_dim();

    let k_row_alt = unit / n;
    let col = unit % n;

    let num_lines = n / rhs_line_size;
    let col_idx = col / rhs_line_size;
    let line_idx = col % rhs_line_size;

    let row_jump = plane_dim / n;

    #[unroll]
    for k_iter in 0..k / row_jump {
        let k_row = row_jump * k_iter + k_row_alt;
        let offset = k_row * num_lines + col_idx;
        let line = slice_from[offset];

        slice_to[k_iter] = if comptime!(rhs_line_size == 1) {
            E::cast_from(line)
        } else {
            line[line_idx]
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
    #[comptime] config: Config,
) {
    let rhs_line_size = config.line_size(Ident::Rhs);
    let plane_dim = config.plane_dim();

    let k_row_alt = unit / n;
    let col = unit % n;
    let row_jump = plane_dim / n;
    let col_offset = col * k / rhs_line_size;

    #[unroll]
    for k_iter in 0..k / row_jump {
        let row = row_jump * k_iter + k_row_alt;
        let row_index = row / rhs_line_size;
        let offset = row_index + col_offset;
        let line = slice_from[offset];
        slice_to[k_iter] = if comptime!(rhs_line_size == 1) {
            E::cast_from(line)
        } else {
            let line_idx = row % rhs_line_size;
            line[line_idx]
        }
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
    plane_mma_line_size: u32,
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
            plane_mma_line_size: 1,
        }
    }
}
