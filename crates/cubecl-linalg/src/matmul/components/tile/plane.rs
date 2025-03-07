use crate::matmul::components::config::MatmulConfig;
use crate::matmul::components::{
    tile, Ident, InvalidConfigError, MatmulConfigFactory, MatmulPrecision, MatrixLayout,
};
use crate::matmul::components::{MatmulProblem, MatmulSize};
use crate::matmul::kernels::MatmulAvailabilityError;
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, Feature};
use tile::{TileConfig, TileMatmul, TileMatmulFamily};

use super::Tile;

/// PlaneMMA instruction uses plane cooperation but does not rely on tensor cores
///
/// # Note
///
/// This is not yet fully optimized
///  - Operations are not vectorized. The algorithm must be rethought to perform operations on lines, perhaps using dot
///  - When loading perpendicular to the lines, too much data is loaded from in comparison to what is used.
///    A solution could be to always load the stage with lhs in row-major and rhs in col-major, using only parallel fill
///  - If not vec4, there are patches in read_output that may harm performance
pub struct PlaneMma;

impl TileMatmulFamily for PlaneMma {
    type Matmul<I: Numeric, O: Numeric> = Self;

    fn tile_shape(config: &Self::Config) -> MatmulSize {
        config.size
    }

    fn requires_tensor_cores() -> bool {
        false
    }
}

#[cube]
impl<I: Numeric, O: Numeric> TileMatmul<I, O> for PlaneMma {
    type Config = Config;
    type Lhs = Array<I>;
    type Rhs = Array<I>;
    type Accumulator = Array<O>;

    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        out: &mut Self::Accumulator,
        #[comptime] config: Config,
    ) {
        let k_jump = config.plane_dim() / config.size.n;
        let row_division = config.plane_dim() / config.size.m;

        let num_jumps = config.size.k / k_jump;
        let compute_width = config.size.n / row_division;

        let unit_offset = UNIT_POS_X % row_division * compute_width;

        #[unroll]
        for k_outer in 0..num_jumps {
            let b_kp = rhs[k_outer];

            #[unroll]
            for k_inner in 0..k_jump {
                let a_pk = lhs[k_outer * k_jump + k_inner];

                #[unroll]
                for n_iter in 0..compute_width {
                    let unit_to_read = k_inner * config.size.n + n_iter + unit_offset;
                    let b_kn = plane_broadcast::<I>(b_kp, unit_to_read);
                    out[n_iter] += O::cast_from(a_pk * b_kn);
                }
            }
        }
    }

    fn allocate_lhs(#[comptime] config: Config) -> Self::Lhs {
        Array::new(config.size.k)
    }

    fn allocate_rhs(#[comptime] config: Config) -> Self::Rhs {
        Array::new(config.size.k * config.size.n / config.plane_dim())
    }

    fn fill_lhs(tile: &Tile<I>, lhs: &mut Self::Lhs, #[comptime] config: Config) {
        match config.matrix_layout(Ident::Lhs) {
            MatrixLayout::RowMajor => fill_parallel_lhs(
                &tile.slice,
                &mut lhs.to_slice_mut(),
                UNIT_POS_X,
                tile.stride,
                config.size.m,
                config.size.k,
                config.line_size(Ident::Lhs),
                config.plane_dim(),
            ),
            MatrixLayout::ColMajor => fill_perpendicular_lhs(
                &tile.slice,
                &mut lhs.to_slice_mut(),
                UNIT_POS_X,
                tile.stride,
                config.size.m,
                config.size.k,
                config.line_size(Ident::Lhs),
                config.plane_dim(),
            ),
        }
    }

    fn fill_rhs(tile: &Tile<I>, rhs: &mut Self::Rhs, #[comptime] config: Config) {
        match comptime!(config.matrix_layout(Ident::Rhs)) {
            MatrixLayout::RowMajor => fill_perpendicular_rhs(
                &tile.slice,
                &mut rhs.to_slice_mut(),
                UNIT_POS_X,
                tile.stride,
                config.size.n,
                config.size.k,
                config.line_size(Ident::Rhs),
                config.plane_dim(),
            ),
            MatrixLayout::ColMajor => fill_parallel_rhs(
                &tile.slice,
                &mut rhs.to_slice_mut(),
                UNIT_POS_X,
                tile.stride,
                config.size.n,
                config.size.k,
                config.line_size(Ident::Rhs),
                config.plane_dim(),
            ),
        }
    }

    fn fill_accumulator(tile: &Tile<O>, acc: &mut Self::Accumulator, #[comptime] config: Config) {
        let unit = UNIT_POS_X;
        let n = config.size.n;
        let line_size = config.line_size(Ident::Out);
        let plane_dim = config.plane_dim();

        let m_row_alt = unit / n;
        let col = unit % n;

        let col_idx = col / line_size;
        let line_idx = col % line_size;

        let row_jump = plane_dim / n;

        #[unroll]
        for m_iter in 0..config.size.m / row_jump {
            let m_row = row_jump * m_iter + m_row_alt;
            let offset = m_row * tile.stride + col_idx;
            let line = tile.slice[offset];

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

        let row_division = plane_dim / config.size.m;
        let compute_width = config.size.n / row_division;
        let num_lines = compute_width / line_size;

        let unit = UNIT_POS_X;

        let row = unit / row_division;
        let row_offset = row * config.size.n / line_size;

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

    fn allocate_accumulator(#[comptime] config: Config) -> Self::Accumulator {
        let len = config.size.m * config.size.n / (config.plane_dim());
        Array::new(len)
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        let len = config.size.m * config.size.n / (config.plane_dim());

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
    stride: u32,
    #[comptime] m: u32,
    #[comptime] k: u32,
    #[comptime] line_size: u32,
    #[comptime] plane_dim: u32,
) {
    let num_lines = k / line_size;
    let row = unit * m / plane_dim;

    #[unroll]
    for col in 0..num_lines {
        let line = slice_from[row * stride + col];
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
    stride: u32,
    #[comptime] m: u32,
    #[comptime] k: u32,
    #[comptime] line_size: u32,
    #[comptime] plane_dim: u32,
) {
    let row = unit * m / plane_dim;

    let row_idx = row / line_size;
    let line_idx = row % line_size;

    #[unroll]
    for col in 0..k {
        let line = slice_from[row_idx + col * stride];
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
    stride: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
    #[comptime] line_size: u32,
    #[comptime] plane_dim: u32,
) {
    let k_row_alt = unit / n;
    let col = unit % n;

    let col_idx = col / line_size;
    let line_idx = col % line_size;

    let row_jump = plane_dim / n;

    #[unroll]
    for k_iter in 0..k / row_jump {
        let k_row = row_jump * k_iter + k_row_alt;
        let offset = k_row * stride + col_idx;
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
    stride: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
    #[comptime] line_size: u32,
    #[comptime] plane_dim: u32,
) {
    let k_row_alt = unit / n;
    let col = unit % n;
    let row_jump = plane_dim / n;
    let col_offset = col * stride;

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

impl MatmulConfigFactory for PlaneMma {
    type Config = Config;
    type Input = MatmulSize;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        if config.size.m * config.size.n % config.plane_dim != 0 {
            return Err(Box::new("Todo"));
        }

        if config.size.k * config.size.n % config.plane_dim != 0 {
            return Err(Box::new("Todo"));
        }

        Ok(())
    }

    fn make_config(
        input: Self::Input,
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        _cube_count: &CubeCount,
        _quantized: bool,
    ) -> Self::Config {
        Config::new(
            input,
            cube_dim.x,
            problem.lhs_layout,
            problem.rhs_layout,
            problem.lhs_line_size as u32,
            problem.rhs_line_size as u32,
            problem.out_line_size as u32,
        )
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        _config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        let i_elem = MP::EG::as_elem_native_unchecked();
        let o_elem = MP::EG::as_elem_native_unchecked();

        if !client.properties().feature_enabled(Feature::Plane) {
            return Err(MatmulAvailabilityError::PlaneOperationsUnavailable);
        }

        if !(client.properties().feature_enabled(Feature::Type(i_elem))
            && client.properties().feature_enabled(Feature::Type(o_elem)))
        {
            return Err(MatmulAvailabilityError::TypesUnavailable {
                input: i_elem,
                output: o_elem,
            });
        }

        Ok(())
    }
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for PlaneMma instruction
pub struct Config {
    size: MatmulSize,
    plane_dim: u32,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    lhs_line_size: u32,
    rhs_line_size: u32,
    out_line_size: u32,
}

impl TileConfig for Config {
    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }

    fn matrix_layout(&self, ident: Ident) -> MatrixLayout {
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

    fn tile_shape(&self) -> &MatmulSize {
        &self.size
    }
}

impl MatmulConfig for Config {}

impl Config {
    pub fn new(
        size: MatmulSize,
        plane_dim: u32,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        lhs_line_size: u32,
        rhs_line_size: u32,
        out_line_size: u32,
    ) -> Self {
        Self {
            size,
            plane_dim,
            lhs_layout,
            rhs_layout,
            lhs_line_size,
            rhs_line_size,
            out_line_size,
        }
    }
}
