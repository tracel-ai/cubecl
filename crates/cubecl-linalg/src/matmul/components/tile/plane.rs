use crate::matmul::components::config::MatmulConfig;
use crate::matmul::components::tile::Config as TileConfig;
use crate::matmul::components::MatmulProblem;
use crate::matmul::components::{tile, Ident, MatmulKernel, MatrixLayout};
use crate::matmul::kernels::matmul::AdvancedConfig;
use crate::matmul::kernels::MatmulAvailabilityError;
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, Feature};
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
///  - Operations are not vectorized. The algorithm must be rethought to perform operations on lines, perhaps using dot
///  - When loading perpendicular to the lines, too much data is loaded from in comparison to what is used.
///    A solution could be to always load the stage with lhs in row-major and rhs in col-major, using only parallel fill
///  - If not vec4, there are patches in read_output that may harm performance
pub struct PlaneMma<I: Numeric, O: Numeric, const M: u32, const N: u32, const K: u32> {
    _input: PhantomData<I>,
    _output: PhantomData<O>,
}

#[cube]
impl<I: Numeric, O: Numeric, const M: u32, const N: u32, const K: u32> tile::Matmul<I, O>
    for PlaneMma<I, O, M, N, K>
{
    const M: u32 = M;
    const N: u32 = N;
    const K: u32 = K;

    type Lhs = Array<I>;
    type Rhs = Array<I>;
    type Accumulator = Array<O>;

    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        out: &mut Self::Accumulator,
        #[comptime] config: Config,
    ) {
        let k_jump = config.plane_dim() / Self::N;
        let row_division = config.plane_dim() / Self::M;

        let num_jumps = Self::K / k_jump;
        let compute_width = Self::N / row_division;

        let unit_offset = UNIT_POS_X % row_division * compute_width;

        #[unroll]
        for k_outer in 0..num_jumps {
            let b_kp = rhs[k_outer];

            #[unroll]
            for k_inner in 0..k_jump {
                let a_pk = lhs[k_outer * k_jump + k_inner];

                #[unroll]
                for n_iter in 0..compute_width {
                    let unit_to_read = k_inner * Self::N + n_iter + unit_offset;
                    let b_kn = plane_broadcast::<I>(b_kp, unit_to_read);
                    out[n_iter] += O::cast_from(a_pk * b_kn);
                }
            }
        }
    }

    fn init_lhs(#[comptime] _config: Config) -> Self::Lhs {
        Array::new(Self::K)
    }

    fn init_rhs(#[comptime] config: Config) -> Self::Rhs {
        Array::new(Self::K * Self::N / config.plane_dim())
    }

    fn fill_lhs(slice: &Slice<Line<I>>, lhs: &mut Self::Lhs, #[comptime] config: Config) {
        match config.layout(Ident::Lhs) {
            MatrixLayout::RowMajor => fill_parallel_lhs(
                slice,
                &mut lhs.to_slice_mut(),
                UNIT_POS_X,
                Self::M,
                Self::K,
                config.line_size(Ident::Lhs),
                config.plane_dim(),
            ),
            MatrixLayout::ColMajor => fill_perpendicular_lhs(
                slice,
                &mut lhs.to_slice_mut(),
                UNIT_POS_X,
                Self::M,
                Self::K,
                config.line_size(Ident::Lhs),
                config.plane_dim(),
            ),
        }
    }

    fn fill_rhs(slice: &Slice<Line<I>>, rhs: &mut Self::Rhs, #[comptime] config: Config) {
        match comptime!(config.layout(Ident::Rhs)) {
            MatrixLayout::RowMajor => fill_perpendicular_rhs(
                slice,
                &mut rhs.to_slice_mut(),
                UNIT_POS_X,
                Self::N,
                Self::K,
                config.line_size(Ident::Rhs),
                config.plane_dim(),
            ),
            MatrixLayout::ColMajor => fill_parallel_rhs(
                slice,
                &mut rhs.to_slice_mut(),
                UNIT_POS_X,
                Self::N,
                Self::K,
                config.line_size(Ident::Rhs),
                config.plane_dim(),
            ),
        }
    }

    fn fill_accumulator(
        slice: &Slice<Line<O>>,
        acc: &mut Self::Accumulator,
        stride: u32,
        #[comptime] config: Config,
    ) {
        let unit = UNIT_POS_X;
        let n = Self::N;
        let line_size = config.line_size(Ident::Out);
        let plane_dim = config.plane_dim();

        let m_row_alt = unit / n;
        let col = unit % n;

        let num_lines = stride / line_size;
        let col_idx = col / line_size;
        let line_idx = col % line_size;

        let row_jump = plane_dim / n;

        #[unroll]
        for m_iter in 0..Self::M / row_jump {
            let m_row = row_jump * m_iter + m_row_alt;
            let offset = m_row * num_lines + col_idx;
            let line = slice[offset];

            acc[m_iter] = if comptime!(line_size == 1) {
                O::cast_from(line)
            } else {
                line[line_idx]
            };
        }
    }

    fn read_accumulator<C: Numeric>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<C>>,
        #[comptime] config: Config,
    ) {
        let line_size = config.line_size(Ident::Out);
        let plane_dim = config.plane_dim();

        let row_division = plane_dim / Self::M;
        let compute_width = Self::N / row_division;
        let num_lines = compute_width / line_size;

        let unit = UNIT_POS_X;

        let row = unit / row_division;
        let row_offset = row * Self::N / line_size;

        let offset = row_offset + unit % row_division * num_lines;

        if comptime!(line_size >= 4) {
            #[unroll]
            for col in 0..num_lines {
                let mut line = Line::<C>::empty(line_size);
                #[unroll]
                for j in 0..line_size {
                    line[j] = C::cast_from(out[col * line_size + j]);
                }
                slice[col + offset] = line;
            }
        } else {
            // There are weird behaviours on wgpu on metal with vec1 and vec2,
            // where some values are left empty.
            // This is patched by repeating loops or deactivating unrolling

            if comptime!(line_size == 1) {
                #[unroll]
                for col in 0..num_lines {
                    slice[col + offset] = Line::cast_from(out[col]);
                }
                // It seems we must repeat this loop without unroll
                for col in 0..num_lines {
                    slice[col + offset] = Line::cast_from(out[col]);
                }
            } else {
                // Cannot be unrolled, leads to bugs
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

    fn init_accumulator(#[comptime] config: Config) -> Self::Accumulator {
        let len = Self::M * Self::N / (config.plane_dim());
        Array::new(len)
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        let len = Self::M * Self::N / (config.plane_dim());

        #[unroll]
        for i in 0..len {
            acc[i] = O::from_int(0);
        }
    }
}

#[cube]
fn fill_parallel_lhs<E: Numeric>(
    slice_from: &Slice<Line<E>>,
    slice_to: &mut SliceMut<E>,
    unit: u32,
    #[comptime] m: u32,
    #[comptime] k: u32,
    #[comptime] line_size: u32,
    #[comptime] plane_dim: u32,
) {
    let num_lines = k / line_size;
    let row = unit * m / plane_dim;

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
    slice_from: &Slice<Line<E>>,
    slice_to: &mut SliceMut<E>,
    unit: u32,
    #[comptime] m: u32,
    #[comptime] k: u32,
    #[comptime] line_size: u32,
    #[comptime] plane_dim: u32,
) {
    let num_lines = m / line_size;
    let row = unit * m / plane_dim;

    let row_idx = row / line_size;
    let line_idx = row % line_size;

    #[unroll]
    for col in 0..k {
        let line = slice_from[row_idx + col * num_lines];
        slice_to[col] = if comptime!(line_size == 1) {
            E::cast_from(line)
        } else {
            line[line_idx]
        };
    }
}

#[cube]
fn fill_perpendicular_rhs<E: Numeric>(
    slice_from: &Slice<Line<E>>,
    slice_to: &mut SliceMut<E>,
    unit: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
    #[comptime] line_size: u32,
    #[comptime] plane_dim: u32,
) {
    let k_row_alt = unit / n;
    let col = unit % n;

    let num_lines = n / line_size;
    let col_idx = col / line_size;
    let line_idx = col % line_size;

    let row_jump = plane_dim / n;

    #[unroll]
    for k_iter in 0..k / row_jump {
        let k_row = row_jump * k_iter + k_row_alt;
        let offset = k_row * num_lines + col_idx;
        let line = slice_from[offset];

        slice_to[k_iter] = if comptime!(line_size == 1) {
            E::cast_from(line)
        } else {
            line[line_idx]
        };
    }
}

#[cube]
fn fill_parallel_rhs<E: Numeric>(
    slice_from: &Slice<Line<E>>,
    slice_to: &mut SliceMut<E>,
    unit: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
    #[comptime] line_size: u32,
    #[comptime] plane_dim: u32,
) {
    let k_row_alt = unit / n;
    let col = unit % n;
    let row_jump = plane_dim / n;
    let col_offset = col * k / line_size;

    #[unroll]
    for k_iter in 0..k / row_jump {
        let row = row_jump * k_iter + k_row_alt;
        let row_index = row / line_size;
        let offset = row_index + col_offset;
        let line = slice_from[offset];
        slice_to[k_iter] = if comptime!(line_size == 1) {
            E::cast_from(line)
        } else {
            let line_idx = row % line_size;
            line[line_idx]
        }
    }
}

impl<I: Numeric, O: Numeric, const M: u32, const N: u32, const K: u32> MatmulKernel
    for PlaneMma<I, O, M, N, K>
{
    type Config = Config;

    fn check_config(config: Self::Config) {
        let plane_dim = config.plane_dim();
        assert!(M * N % plane_dim == 0);
        assert!(K * N % plane_dim == 0);
    }

    fn check_availability<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<(), MatmulAvailabilityError> {
        if !client.properties().feature_enabled(Feature::Plane) {
            return Err(MatmulAvailabilityError::PlaneOperationsUnavailable);
        }

        if !(client
            .properties()
            .feature_enabled(Feature::Type(I::as_elem()))
            && client
                .properties()
                .feature_enabled(Feature::Type(O::as_elem())))
        {
            return Err(MatmulAvailabilityError::TypesUnavailable {
                input: I::as_elem(),
                output: O::as_elem(),
            });
        }

        Ok(())
    }

    fn make_config(
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        _cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Self::Config {
        let (lhs_tile_layout, lhs_tile_line_size) = match advanced_config.enforced_tile_layout.0 {
            Some(enforced_layout) if enforced_layout != problem.lhs_layout => (enforced_layout, 1),
            _ => (problem.lhs_layout, problem.lhs_line_size),
        };

        let (rhs_tile_layout, rhs_tile_line_size) = match advanced_config.enforced_tile_layout.1 {
            Some(enforced_layout) if enforced_layout != problem.rhs_layout => (enforced_layout, 1),
            _ => (problem.rhs_layout, problem.rhs_line_size),
        };

        Config::new(
            cube_dim.x,
            lhs_tile_layout,
            rhs_tile_layout,
            lhs_tile_line_size as u32,
            rhs_tile_line_size as u32,
            problem.out_line_size as u32,
        )
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
        }
    }
}
