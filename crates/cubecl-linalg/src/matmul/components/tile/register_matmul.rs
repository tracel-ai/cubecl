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
use cubecl_std::CubeOption;

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

#[derive(CubeType)]
/// Contains the accumulated result, within a row major array of size rows x cols
pub struct TileAccumulator<EA: Numeric> {
    data: Array<Line<EA>>,
    #[cube(comptime)]
    rows: u32,
    #[cube(comptime)]
    line_cols: u32,
}

#[derive(CubeType)]
pub struct TileInput<ES: Numeric> {
    tile: ComptimeCell<CubeOption<Tile<ES>>>,
}

#[cube]
impl<MP: MatmulPrecision> TileMatmul<MP> for RegisterMatmul {
    type Config = Config;
    type Lhs = TileInput<MP::ES>;
    type Rhs = TileInput<MP::ES>;
    type Accumulator = TileAccumulator<MP::EA>;

    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        out: &mut Self::Accumulator,
        #[comptime] config: Config,
    ) {
        let m = config.size.m;
        let n = config.size.n;
        let k = config.size.k;

        let lhs = lhs.tile.read().unwrap();
        let rhs = rhs.tile.read().unwrap();

        // TODO
        // - If line < m,k, iterate on lines
        // - Implement for other layouts. If in the end they all look the same, refactor. Otherwise not a big deal
        // Don't make loaders that transpose, not worth it at all.

        match comptime!((lhs.layout, rhs.layout)) {
            (MatrixLayout::RowMajor, MatrixLayout::RowMajor) => unimplemented!(),
            (MatrixLayout::RowMajor, MatrixLayout::ColMajor) => unimplemented!(),
            (MatrixLayout::ColMajor, MatrixLayout::RowMajor) => {
                assert!(config.size.m == config.lhs_line_size);
                assert!(config.size.n == config.rhs_line_size);
                #[unroll]
                for k_ in 0..k {
                    let l: Line<MP::EA> = Line::cast_from(lhs.get_line(0, k_));
                    let r: Line<MP::EA> = Line::cast_from(rhs.get_line(k_, 0));
                    #[unroll]
                    for m_ in 0..m {
                        out.data[m_] += Line::cast_from(l[m_]) * r;
                    }
                }
            }
            (MatrixLayout::ColMajor, MatrixLayout::ColMajor) => unimplemented!(),
        }
    }

    fn allocate_lhs(#[comptime] _config: Config) -> Self::Lhs {
        TileInput::<MP::ES> {
            tile: ComptimeCell::new(CubeOption::new_None()),
        }
    }

    fn allocate_rhs(#[comptime] _config: Config) -> Self::Rhs {
        TileInput::<MP::ES> {
            tile: ComptimeCell::new(CubeOption::new_None()),
        }
    }

    fn fill_lhs(tile: &Tile<MP::ES>, lhs: &mut Self::Lhs, #[comptime] _config: Config) {
        lhs.tile
            .store(CubeOption::new_Some(comptime!(tile.clone())));
    }

    fn fill_rhs(tile: &Tile<MP::ES>, rhs: &mut Self::Rhs, #[comptime] _config: Config) {
        rhs.tile
            .store(CubeOption::new_Some(comptime!(tile.clone())));
    }

    fn fill_accumulator(
        tile: &Tile<MP::EA>,
        acc: &mut Self::Accumulator,
        #[comptime] _config: Config,
    ) {
        #[unroll]
        for i in 0..comptime!(acc.rows * acc.line_cols) {
            acc.data[i] = tile.slice[i * tile.stride];
        }
    }

    fn write_results(
        acc: &Self::Accumulator,
        slice: &mut SliceMut<Line<MP::EO>>,
        #[comptime] _config: Config,
    ) {
        // Assuming acc and slice have same line size
        #[unroll]
        for i in 0..comptime!(acc.rows * acc.line_cols) {
            slice[i] = Line::cast_from(acc.data[i]);
        }
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let rows = config.size.m;
        let line_cols = config.size.n / config.out_line_size;

        TileAccumulator::<MP::EA> {
            data: Array::<Line<MP::EA>>::vectorized(rows * line_cols, config.out_line_size),
            rows,
            line_cols,
        }
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] _config: Self::Config) {
        #[unroll]
        for i in 0..comptime!(acc.rows * acc.line_cols) {
            acc.data[i] = Line::cast_from(0);
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
