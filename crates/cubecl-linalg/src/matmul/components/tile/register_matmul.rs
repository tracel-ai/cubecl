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
/// # Assumptions
/// - The array has a length of `tile_size`.
/// - In row major layout, this represents `tile_size` rows of 1 line each (`tile_size x 1`).
/// - In column major layout, this represents `tile_size` columns of 1 line each (`1 x tile_size`).
///
/// Both layouts are supported, but performance is typically better with:
/// - Lhs in column-major layout
/// - Rhs in row-major layout
#[derive(CubeType)]
pub struct TileArray<E: Numeric> {
    data: Array<Line<E>>,
    #[cube(comptime)]
    layout: MatrixLayout,
    #[cube(comptime)]
    tile_size: u32,
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
        let tile_size = lhs.tile_size;

        match comptime!(lhs.layout) {
            MatrixLayout::RowMajor => match comptime!(rhs.layout) {
                MatrixLayout::RowMajor => {
                    unimplemented!()
                }
                MatrixLayout::ColMajor => {
                    unimplemented!()
                }
            },
            MatrixLayout::ColMajor => match comptime!(rhs.layout) {
                MatrixLayout::RowMajor => {
                    #[unroll]
                    for k in 0..tile_size {
                        // Get kth line from Lhs, keep in ES, it will be converted to EA individually
                        let line_lhs = lhs.data[k];
                        // Get kth line from Rhs, in EA
                        let line_rhs: Line<MP::EA> = Line::cast_from(rhs.data[k]);

                        // Break the lines of lhs and broadcast each element on rhs
                        for m in 0..tile_size {
                            out.data[m] += Line::cast_from(line_lhs[m]) * line_rhs;
                        }
                    }
                }
                MatrixLayout::ColMajor => {
                    unimplemented!()
                }
            },
        }

        // let lhs_data: Slice<MP::EA> = lhs.data.to_slice().try_cast_unchecked();
        // let rhs_data: Slice<MP::EA> = rhs.data.to_slice().try_cast_unchecked();
        // let mut out_data: SliceMut<MP::EA> = out.data.to_slice_mut().try_cast_unchecked();

        // #[unroll]
        // for k in 0..tile_size {
        //     #[unroll]
        //     for m in 0..tile_size {
        //         let lhs_index = match comptime!(lhs.layout) {
        //             MatrixLayout::RowMajor => k * tile_size + m,
        //             MatrixLayout::ColMajor => m * tile_size + k,
        //         };
        //         let lhs_val = MP::EA::cast_from(lhs_data[lhs_index]);

        //         #[unroll]
        //         for n in 0..tile_size {
        //             let rhs_index = match comptime!(rhs.layout) {
        //                 MatrixLayout::RowMajor => n * tile_size + k,
        //                 MatrixLayout::ColMajor => k * tile_size + n,
        //             };
        //             let rhs_val = MP::EA::cast_from(rhs_data[rhs_index]);

        //             let out_index = m * tile_size + n;

        //             // Add assign not supported on slices?
        //             let mut out_elem = out_data[out_index];
        //             out_elem += lhs_val * rhs_val;
        //             out_data[out_index] = out_elem;
        //         }
        //     }
        // }
    }

    fn allocate_lhs(#[comptime] config: Config) -> Self::Lhs {
        TileArray::<MP::ES> {
            data: Array::<Line<MP::ES>>::new(config.tile_size),
            layout: config.lhs_layout,
            tile_size: config.tile_size,
        }
    }

    fn allocate_rhs(#[comptime] config: Config) -> Self::Rhs {
        TileArray::<MP::ES> {
            data: Array::<Line<MP::ES>>::new(config.tile_size),
            layout: config.rhs_layout,
            tile_size: config.tile_size,
        }
    }

    fn fill_lhs(tile: &Tile<MP::ES>, lhs: &mut Self::Lhs, #[comptime] _config: Config) {
        #[unroll]
        for i in 0..lhs.tile_size {
            lhs.data[i] = tile.slice[i * tile.stride];
        }
    }

    fn fill_rhs(tile: &Tile<MP::ES>, rhs: &mut Self::Rhs, #[comptime] _config: Config) {
        #[unroll]
        for i in 0..rhs.tile_size {
            rhs.data[i] = tile.slice[i * tile.stride];
        }
    }

    fn fill_accumulator(
        tile: &Tile<MP::EA>,
        acc: &mut Self::Accumulator,
        #[comptime] _config: Config,
    ) {
        #[unroll]
        for i in 0..acc.tile_size {
            acc.data[i] = tile.slice[i * tile.stride];
        }
    }

    fn read_accumulator<C: Numeric>(
        acc: &Self::Accumulator,
        slice: &mut SliceMut<Line<C>>,
        #[comptime] _config: Config,
    ) {
        #[unroll]
        for i in 0..acc.tile_size {
            slice[i] = Line::cast_from(acc.data[i]);
        }
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let tile_size = config.stage_line_size(Ident::Out);

        TileArray::<MP::EA> {
            data: Array::<Line<MP::EA>>::new(tile_size),
            layout: MatrixLayout::RowMajor,
            tile_size,
        }
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] _config: Self::Config) {
        #[unroll]
        for i in 0..acc.tile_size {
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
        let lhs = config.stage_line_size(Ident::Lhs);
        let rhs = config.stage_line_size(Ident::Rhs);
        let out = config.stage_line_size(Ident::Out);

        if !(m == n && n == k && k == lhs && lhs == rhs && rhs == out) {
            return Err(RegisterMatmulConfigError::new(move || {
                format!(
                    "Register matmul needs config m ({:?}), n ({:?}), k ({:?}), line size for lhs ({:?}), rhs ({:?}) and out ({:?}) to be the same.",
                    m, n, k, lhs, rhs, out
                )
            }));
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
