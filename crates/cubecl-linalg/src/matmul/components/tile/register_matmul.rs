use std::fmt::Display;

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

/// Wrapper over an array representing a tile of data.
///
/// TODO: data is not lined, because lined array support seems flaky
#[derive(CubeType)]
pub struct TileArray<E: Numeric> {
    data: Array<E>,
    #[cube(comptime)]
    layout: MatrixLayout,
    #[cube(comptime)]
    rows: u32,
    #[cube(comptime)]
    cols: u32,
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
        #[comptime] _config: Config,
    ) {
        match comptime!(lhs.layout) {
            MatrixLayout::RowMajor => match comptime!(rhs.layout) {
                MatrixLayout::RowMajor =>
                {
                    #[unroll]
                    for k_ in 0..lhs.cols {
                        #[unroll]
                        for i in 0..lhs.rows {
                            let l = MP::EA::cast_from(lhs.data[i * lhs.cols + k_]);
                            #[unroll]
                            for j in 0..rhs.cols {
                                let r = MP::EA::cast_from(rhs.data[k_ * rhs.cols + j]);
                                let o = i * rhs.cols + j;
                                out.data[o] = out.data[o] + l * r;
                            }
                        }
                    }
                }
                MatrixLayout::ColMajor =>
                {
                    #[unroll]
                    for k_ in 0..lhs.cols {
                        #[unroll]
                        for i in 0..lhs.rows {
                            let l = MP::EA::cast_from(lhs.data[i * lhs.cols + k_]);
                            #[unroll]
                            for j in 0..rhs.cols {
                                let r = MP::EA::cast_from(rhs.data[j * rhs.rows + k_]);
                                let o = i * rhs.cols + j;
                                out.data[o] = out.data[o] + l * r;
                            }
                        }
                    }
                }
            },
            MatrixLayout::ColMajor => match comptime!(rhs.layout) {
                MatrixLayout::RowMajor =>
                {
                    #[unroll]
                    for k_ in 0..lhs.cols {
                        #[unroll]
                        for i in 0..lhs.rows {
                            let l = MP::EA::cast_from(lhs.data[k_ * lhs.rows + i]);
                            #[unroll]
                            for j in 0..rhs.cols {
                                let r = MP::EA::cast_from(rhs.data[k_ * rhs.cols + j]);
                                let o = i * rhs.cols + j;
                                out.data[o] = out.data[o] + l * r;
                            }
                        }
                    }
                }
                MatrixLayout::ColMajor =>
                {
                    #[unroll]
                    for k_ in 0..lhs.cols {
                        #[unroll]
                        for i in 0..lhs.rows {
                            let l = MP::EA::cast_from(lhs.data[k_ * lhs.rows + i]);
                            #[unroll]
                            for j in 0..rhs.cols {
                                let r = MP::EA::cast_from(rhs.data[j * rhs.rows + k_]);
                                let o = i * rhs.cols + j;
                                out.data[o] = out.data[o] + l * r;
                            }
                        }
                    }
                }
            },
        }
    }

    fn allocate_lhs(#[comptime] config: Config) -> Self::Lhs {
        let rows = config.tile_size;
        let cols = config.tile_size;
        TileArray::<MP::ES> {
            data: Array::<MP::ES>::new(rows * cols),
            layout: config.lhs_layout,
            rows,
            cols,
        }
    }

    fn allocate_rhs(#[comptime] config: Config) -> Self::Rhs {
        let rows = config.tile_size;
        let cols = config.tile_size;
        TileArray::<MP::ES> {
            data: Array::<MP::ES>::new(rows * cols),
            layout: config.rhs_layout,
            rows,
            cols,
        }
    }

    fn fill_lhs(tile: &Tile<MP::ES>, lhs: &mut Self::Lhs, #[comptime] config: Config) {
        let line_size = config.lhs_line_size;

        match comptime!(lhs.layout) {
            MatrixLayout::RowMajor => {
                let num_lines_per_row = comptime!(lhs.cols / line_size); // assumed to be perfectly divisible as per check config
                #[unroll]
                for row in 0..lhs.rows {
                    #[unroll]
                    for r in 0..num_lines_per_row {
                        let line = tile.slice[row * num_lines_per_row + r];
                        #[unroll]
                        for i in 0..line_size {
                            lhs.data[row * lhs.cols + r * line_size + i] = line[i];
                        }
                    }
                }
            }
            MatrixLayout::ColMajor => {
                let num_lines_per_col = comptime!(lhs.rows / line_size); // assumed to be perfectly divisible as per check config
                #[unroll]
                for col in 0..lhs.cols {
                    #[unroll]
                    for l in 0..num_lines_per_col {
                        let line = tile.slice[col * num_lines_per_col + l];
                        #[unroll]
                        for i in 0..line_size {
                            lhs.data[col * lhs.rows + l * line_size + i] = line[i];
                        }
                    }
                }
            }
        }
    }

    fn fill_rhs(tile: &Tile<MP::ES>, rhs: &mut Self::Rhs, #[comptime] config: Config) {
        let line_size = config.rhs_line_size;

        match comptime!(rhs.layout) {
            MatrixLayout::RowMajor => {
                let num_lines_per_row = comptime!(rhs.cols / line_size); // assumed to be perfectly divisible as per check config
                #[unroll]
                for row in 0..rhs.rows {
                    #[unroll]
                    for r in 0..num_lines_per_row {
                        let line = tile.slice[row * num_lines_per_row + r];
                        #[unroll]
                        for i in 0..line_size {
                            rhs.data[row * rhs.cols + r * line_size + i] = line[i];
                        }
                    }
                }
            }
            MatrixLayout::ColMajor => {
                let num_lines_per_col = comptime!(rhs.rows / line_size); // assumed to be perfectly divisible as per check config
                #[unroll]
                for col in 0..rhs.cols {
                    #[unroll]
                    for l in 0..num_lines_per_col {
                        let line = tile.slice[col * num_lines_per_col + l];
                        #[unroll]
                        for i in 0..line_size {
                            rhs.data[col * rhs.rows + l * line_size + i] = line[i];
                        }
                    }
                }
            }
        }
    }

    fn fill_accumulator(
        tile: &Tile<MP::EA>,
        acc: &mut Self::Accumulator,
        #[comptime] _config: Config,
    ) {
        #[unroll]
        for i in 0..comptime!(acc.rows * acc.cols) {
            acc.data[i] = tile.slice.with_line_size(1u32)[i * tile.stride][0];
        }
    }

    fn read_accumulator<C: Numeric>(
        acc: &Self::Accumulator,
        slice: &mut SliceMut<Line<C>>,
        #[comptime] config: Config,
    ) {
        let line_size = config.out_line_size;
        let num_lines_per_row = comptime!(acc.cols / line_size);
        #[unroll]
        for row in 0..comptime!(acc.rows) {
            #[unroll]
            for r in 0..comptime!(num_lines_per_row) {
                let mut line = Line::empty(line_size);
                #[unroll]
                for i in 0..comptime!(line_size) {
                    line[i] = acc.data[row * acc.rows + r * line_size + i];
                }
                slice[row * num_lines_per_row + r] = Line::cast_from(line);
            }
        }
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let rows = config.tile_size;
        let cols = config.tile_size;

        TileArray::<MP::EA> {
            data: Array::<MP::EA>::new(rows * cols),
            layout: MatrixLayout::RowMajor,
            rows,
            cols,
        }
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] _config: Self::Config) {
        #[unroll]
        for i in 0..comptime!(acc.rows * acc.cols) {
            acc.data[i] = MP::EA::cast_from(0);
        }
    }
}

pub struct RegisterMatmulConfigError {
    func: Box<dyn Fn() -> String>,
}

impl RegisterMatmulConfigError {
    #[allow(clippy::new_ret_no_self)]
    pub fn new<F: Fn() -> String + 'static>(func: F) -> Box<dyn Display> {
        Box::new(Self {
            func: Box::new(func),
        })
    }
}

impl Display for RegisterMatmulConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = (self.func)();
        write!(f, "{string}")
    }
}
impl MatmulConfigFactory for RegisterMatmul {
    type Input = TileMatmulConfigInput;
    type Config = Config;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        let m = config.size.m;
        let n = config.size.n;
        let k = config.size.k;

        if m > 8 || n > 8 || k > 8 {
            return Err(RegisterMatmulConfigError::new(move || {
                format!(
                    "Tile size m-n-k={:?}-{:?}-{:?} is too large for register matmul",
                    m, n, k
                )
            }));
        }

        let lhs = config.stage_line_size(Ident::Lhs);
        let rhs = config.stage_line_size(Ident::Rhs);

        match config.lhs_layout {
            MatrixLayout::RowMajor => {
                if k % lhs != 0 {
                    return Err(RegisterMatmulConfigError::new(move || {
                        format!(
                            "When Lhs is row major, register matmul k {:?} must be divisible by lhs line size {:?}.",
                            k, lhs
                        )
                    }));
                }
            }
            MatrixLayout::ColMajor => {
                if m % lhs != 0 {
                    return Err(RegisterMatmulConfigError::new(move || {
                        format!(
                            "When Lhs is col major, register matmul m {:?} must be divisible by lhs line size {:?}.",
                            k, lhs
                        )
                    }));
                }
            }
        }

        match config.rhs_layout {
            MatrixLayout::RowMajor => {
                if n % rhs != 0 {
                    return Err(RegisterMatmulConfigError::new(move || {
                        format!(
                            "When Rhs is row major, register matmul n {:?} must be divisible by rhs line size {:?}.",
                            n, rhs
                        )
                    }));
                }
            }
            MatrixLayout::ColMajor => {
                if k % rhs != 0 {
                    return Err(RegisterMatmulConfigError::new(move || {
                        format!(
                            "When Rhs is col major, register matmul k {:?} must be divisible by rhs line size {:?}.",
                            k, rhs
                        )
                    }));
                }
            }
        }

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
    pub tile_size: u32,
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
            tile_size: size.m,
        }
    }
}
