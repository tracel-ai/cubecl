use crate::matmul::components::config::MatmulConfig;
use crate::matmul::components::tile::{TileConfig, TileMatmul, TileMatmulFamily};
use crate::matmul::components::{
    Ident, InvalidConfigError, MatmulConfigFactory, MatmulPrecision, MatmulProblem, MatmulSize,
    MatrixLayout,
};
use crate::matmul::kernels::MatmulAvailabilityError;
use cubecl_core::ir::{Elem, FloatKind};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, Feature};

use super::{Tile, TileMatmulConfigInput};

/// Uses one unit to perform a small matmul entirely using its registers
pub struct RegisterMatmul;

impl TileMatmulFamily for RegisterMatmul {
    type Matmul<MP: MatmulPrecision> = RegisterMatmul;

    fn tile_shape(config: &Self::Config) -> MatmulSize {
        config.size
    }

    fn requires_tensor_cores() -> bool {
        false
    }
}

/// Wrapper over an array to represent a tile
#[derive(CubeType)]
pub struct TileArray<E: Numeric> {
    data: Array<Line<E>>,
    #[cube(comptime)]
    layout: MatrixLayout,
    #[cube(comptime)]
    row_count: u32,
    #[cube(comptime)]
    col_count: u32,
}

#[cube]
impl<MP: MatmulPrecision> TileMatmul<MP> for RegisterMatmul {
    type Config = Config;
    type Lhs = TileArray<MP::ES>;
    type Rhs = TileArray<MP::ES>;
    type Accumulator = TileArray<MP::EA>;

    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        out: &mut Self::Accumulator,
        #[comptime] config: Config,
    ) {
        let m = config.size.m;
        let n = config.size.n;
        let k = config.size.k;

        let lhs_data: Slice<MP::EA> = lhs.data.to_slice().try_cast_unchecked();
        let rhs_data: Slice<MP::EA> = rhs.data.to_slice().try_cast_unchecked();
        let mut out_data: SliceMut<MP::EA> = out.data.to_slice_mut().try_cast_unchecked();

        #[unroll]
        for k_iter in 0..k {
            #[unroll]
            for m_iter in 0..m {
                let lhs_index = match comptime!(lhs.layout) {
                    MatrixLayout::RowMajor => k_iter * m + m_iter,
                    MatrixLayout::ColMajor => m_iter * k + k_iter,
                };
                let lhs_val = MP::EA::cast_from(lhs_data[lhs_index]);

                #[unroll]
                for n_iter in 0..n {
                    let rhs_index = match comptime!(rhs.layout) {
                        MatrixLayout::RowMajor => n_iter * k + k_iter,
                        MatrixLayout::ColMajor => k_iter * n + n_iter,
                    };
                    let rhs_val = MP::EA::cast_from(rhs_data[rhs_index]);

                    let out_index = m_iter * n + n_iter;

                    // Add assign not supported on slices?
                    let mut out_elem = out_data[out_index];
                    out_elem += lhs_val * rhs_val;
                    out_data[out_index] = out_elem;
                }
            }
        }
    }

    fn allocate_lhs(#[comptime] config: Config) -> Self::Lhs {
        let m = config.size.m;
        let k = config.size.k;

        TileArray::<MP::ES> {
            data: Array::<Line<MP::ES>>::new(m * k),
            layout: config.lhs_layout,
            row_count: m,
            col_count: k,
        }
    }

    fn allocate_rhs(#[comptime] config: Config) -> Self::Rhs {
        let k = config.size.k;
        let n = config.size.n;

        TileArray::<MP::ES> {
            data: Array::<Line<MP::ES>>::new(k * n),
            layout: config.rhs_layout,
            row_count: k,
            col_count: n,
        }
    }

    fn fill_lhs(tile: &Tile<MP::ES>, lhs: &mut Self::Lhs, #[comptime] config: Config) {
        fill_input(tile, lhs, Ident::Lhs, config);
    }

    fn fill_rhs(tile: &Tile<MP::ES>, rhs: &mut Self::Rhs, #[comptime] config: Config) {
        fill_input(tile, rhs, Ident::Rhs, config);
    }

    fn fill_accumulator(
        tile: &Tile<MP::EA>,
        acc: &mut Self::Accumulator,
        #[comptime] config: Config,
    ) {
        fill_input(tile, acc, Ident::Out, config);
    }

    fn read_accumulator<C: Numeric>(
        acc: &Self::Accumulator,
        slice: &mut SliceMut<Line<C>>,
        #[comptime] config: Config,
    ) {
        let line_size = config.stage_line_size(Ident::Out);

        #[unroll]
        for i in 0..acc.row_count * acc.col_count / line_size {
            slice[i] = Line::cast_from(acc.data[i]);
        }
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let m = config.size.m;
        let n = config.size.n;

        TileArray::<MP::EA> {
            data: Array::<Line<MP::EA>>::new(m * n),
            layout: MatrixLayout::RowMajor,
            row_count: m,
            col_count: n,
        }
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        let line_size = config.stage_line_size(Ident::Out);

        #[unroll]
        for i in 0..acc.row_count * acc.col_count / line_size {
            acc.data[i] = Line::cast_from(0);
        }
    }
}

#[cube]
fn fill_input<E: Numeric>(
    tile: &Tile<E>,
    tile_array: &mut TileArray<E>,
    #[comptime] ident: Ident,
    #[comptime] config: Config,
) {
    let line_size = config.stage_line_size(ident);

    match comptime!(tile_array.layout) {
        MatrixLayout::RowMajor => {
            let row_count = tile_array.row_count;
            let col_count = tile_array.col_count / line_size;

            #[unroll]
            for i in 0..row_count {
                #[unroll]
                for j in 0..col_count {
                    tile_array.data[i * col_count + j] = tile.slice[i * tile.stride + j]
                }
            }
        }
        MatrixLayout::ColMajor => {
            let row_count = tile_array.row_count / line_size;
            let col_count = tile_array.col_count;

            #[unroll]
            for i in 0..col_count {
                #[unroll]
                for j in 0..row_count {
                    tile_array.data[i * row_count + j] = tile.slice[i * tile.stride + j]
                }
            }
        }
    }
}

impl MatmulConfigFactory for RegisterMatmul {
    type Input = TileMatmulConfigInput;
    type Config = Config;

    fn check_config(_config: &Self::Config) -> Result<(), InvalidConfigError> {
        Ok(())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        if config.stage_dynamic_line_size
            && !client
                .properties()
                .feature_enabled(Feature::DynamicLineSize)
        {
            return Err(MatmulAvailabilityError::DynamicLineSizeUnavailable);
        }

        let es = MP::ES::as_elem_native().expect("to be a native type");
        let ea = MP::EA::as_elem_native().expect("to be a native type");

        let es = match es {
            Elem::Float(FloatKind::Flex32) => Elem::Float(FloatKind::F32),
            _ => es,
        };

        let ea = match ea {
            Elem::Float(FloatKind::Flex32) => Elem::Float(FloatKind::F32),
            _ => ea,
        };

        if !(MP::ES::is_supported(client) && MP::EA::is_supported(client)) {
            return Err(MatmulAvailabilityError::TypesUnavailable {
                input: es,
                output: ea,
            });
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
        let (lhs_line_size, rhs_line_size, stage_line_size_update) =
            if input.vectorization.stage_line_size == 0 {
                (
                    problem.lhs_line_size as u32,
                    problem.rhs_line_size as u32,
                    false,
                )
            } else {
                (
                    input.vectorization.stage_line_size as u32,
                    input.vectorization.stage_line_size as u32,
                    true,
                )
            };
        Config::new(
            input.size,
            cube_dim.x,
            problem.lhs_layout,
            problem.rhs_layout,
            stage_line_size_update,
            lhs_line_size,
            rhs_line_size,
            problem.out_line_size as u32,
        )
    }
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for Register instruction
pub struct Config {
    size: MatmulSize,
    plane_dim: u32,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    stage_dynamic_line_size: bool,
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

    fn stage_line_size(&self, ident: Ident) -> u32 {
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        size: MatmulSize,
        plane_dim: u32,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        stage_dynamic_line_size: bool,
        lhs_line_size: u32,
        rhs_line_size: u32,
        out_line_size: u32,
    ) -> Self {
        Self {
            size,
            plane_dim,
            lhs_layout,
            rhs_layout,
            stage_dynamic_line_size,
            lhs_line_size,
            rhs_line_size,
            out_line_size,
        }
    }
}
