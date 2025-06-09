use std::fmt::Display;

use crate::components::config::MatmulConfig;
use crate::components::problem::MatmulLineSizes;
use crate::components::resource::ComputeResources;
use crate::components::tile::{TileConfig, TileMatmul, TileMatmulFamily};
use crate::components::{
    Ident, InvalidConfigError, MatmulConfigFactory, MatmulPrecision, MatmulProblem, MatrixLayout,
    TileSize,
};
use crate::kernels::MatmulAvailabilityError;
use cubecl_core::ir::{Elem, FloatKind};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, Feature};

use super::{Tile, TileMatmulConfigInput};

/// Uses one unit to perform a small matmul entirely using its registers
pub struct RegisterMatmul;

pub enum ProductType {
    /// Needs lhs to be row major and rhs to be col major
    /// If not the case, tile will be transposed
    Inner,
    /// Needs lhs to be col major and rhs to be row major
    /// If not the case, tile will be transposed
    Outer,
}

impl TileMatmulFamily for RegisterMatmul {
    type Matmul<MP: MatmulPrecision> = RegisterMatmul;

    fn requires_tensor_cores() -> bool {
        false
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Units(1))
    }
}

#[derive(CubeType)]
/// Contains the accumulated result, within a row major array of size rows x cols
pub struct TileAccumulator<EA: Numeric> {
    data: Array<EA>,
    #[cube(comptime)]
    rows: u32,
    #[cube(comptime)]
    cols: u32,
}

#[cube]
impl<MP: MatmulPrecision> TileMatmul<MP> for RegisterMatmul {
    type Config = Config;
    type Lhs = Array<MP::ES>;
    type Rhs = Array<MP::ES>;
    type Accumulator = TileAccumulator<MP::EA>;

    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        acc: &mut Self::Accumulator,
        #[comptime] config: Config,
    ) {
        match config.product_type() {
            ProductType::Inner => Self::inner_product(lhs, rhs, acc, config),
            ProductType::Outer => Self::outer_product(lhs, rhs, acc, config),
        }
    }

    fn allocate_lhs(#[comptime] config: Config) -> Self::Lhs {
        Array::new(config.size.mk())
    }

    fn allocate_rhs(#[comptime] config: Config) -> Self::Rhs {
        Array::new(config.size.nk())
    }

    fn fill_lhs(tile: &Tile<MP::ES>, lhs: &mut Self::Lhs, #[comptime] config: Config) {
        match config.product_type() {
            ProductType::Inner => match config.lhs_layout {
                MatrixLayout::RowMajor => {
                    Self::fill_plain(
                        tile,
                        lhs,
                        config.size.m(),
                        config.size.k(),
                        config.lhs_line_size,
                    );
                }
                MatrixLayout::ColMajor => {
                    Self::fill_transposed(
                        tile,
                        lhs,
                        config.size.k(),
                        config.size.m(),
                        config.lhs_line_size,
                    );
                }
            },
            ProductType::Outer => match config.lhs_layout {
                MatrixLayout::RowMajor => {
                    Self::fill_transposed(
                        tile,
                        lhs,
                        config.size.m(),
                        config.size.k(),
                        config.lhs_line_size,
                    );
                }
                MatrixLayout::ColMajor => {
                    Self::fill_plain(
                        tile,
                        lhs,
                        config.size.k(),
                        config.size.m(),
                        config.lhs_line_size,
                    );
                }
            },
        }
    }

    fn fill_rhs(tile: &Tile<MP::ES>, rhs: &mut Self::Rhs, #[comptime] config: Config) {
        match config.product_type() {
            ProductType::Inner => match config.rhs_layout {
                MatrixLayout::RowMajor => {
                    Self::fill_transposed(
                        tile,
                        rhs,
                        config.size.k(),
                        config.size.n(),
                        config.rhs_line_size,
                    );
                }
                MatrixLayout::ColMajor => {
                    Self::fill_plain(
                        tile,
                        rhs,
                        config.size.n(),
                        config.size.k(),
                        config.rhs_line_size,
                    );
                }
            },
            ProductType::Outer => match config.rhs_layout {
                MatrixLayout::RowMajor => {
                    Self::fill_plain(
                        tile,
                        rhs,
                        config.size.k(),
                        config.size.n(),
                        config.rhs_line_size,
                    );
                }
                MatrixLayout::ColMajor => {
                    Self::fill_transposed(
                        tile,
                        rhs,
                        config.size.n(),
                        config.size.k(),
                        config.rhs_line_size,
                    );
                }
            },
        }
    }

    fn fill_accumulator(
        tile: &Tile<MP::EA>,
        acc: &mut Self::Accumulator,
        #[comptime] _config: Config,
    ) {
        #[unroll]
        for i in 0..comptime!(acc.rows) {
            #[unroll]
            for j in 0..comptime!(acc.cols) {
                acc.data[i * acc.cols + j] =
                    tile.slice.with_line_size(1u32)[i * tile.stride + j][0];
            }
        }
    }

    fn write_results(
        acc: &Self::Accumulator,
        slice: &mut SliceMut<Line<MP::EO>>,
        #[comptime] config: Config,
    ) {
        let out_line_size = config.out_line_size;
        #[unroll]
        for i in 0..comptime!(acc.rows * acc.cols / out_line_size) {
            let mut line = Line::empty(out_line_size);
            #[unroll]
            for j in 0..comptime!(out_line_size) {
                line[j] = acc.data[i * out_line_size + j];
            }
            slice[i] = Line::cast_from(line);
        }
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let rows = config.size.m();
        let cols = config.size.n();

        TileAccumulator::<MP::EA> {
            data: Array::<MP::EA>::new(rows * cols),
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

#[cube]
impl RegisterMatmul {
    fn inner_product<ES: Numeric, EA: Numeric>(
        lhs: &Array<ES>,
        rhs: &Array<ES>,
        acc: &mut TileAccumulator<EA>,
        #[comptime] config: Config,
    ) {
        let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = config.size.into(); (m, n, k)};

        #[unroll]
        for m_ in 0..m {
            #[unroll]
            for n_ in 0..n {
                #[unroll]
                for k_ in 0..k {
                    let lhs_elem = EA::cast_from(lhs[m_ * k + k_]);
                    let rhs_elem = EA::cast_from(rhs[n_ * k + k_]);
                    acc.data[m_ * n + n_] += lhs_elem * rhs_elem;
                }
            }
        }
    }

    fn outer_product<ES: Numeric, EA: Numeric>(
        lhs: &Array<ES>,
        rhs: &Array<ES>,
        acc: &mut TileAccumulator<EA>,
        #[comptime] config: Config,
    ) {
        let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = config.size.into(); (m, n, k)};

        #[unroll]
        for k_ in 0..k {
            #[unroll]
            for m_ in 0..m {
                let lhs_elem = EA::cast_from(lhs[k_ * m + m_]);
                #[unroll]
                for n_ in 0..n {
                    let rhs_elem = EA::cast_from(rhs[k_ * n + n_]);
                    acc.data[m_ * n + n_] += lhs_elem * rhs_elem;
                }
            }
        }
    }

    fn fill_plain<ES: Numeric>(
        tile: &Tile<ES>,
        array: &mut Array<ES>,
        #[comptime] num_segments: u32,
        #[comptime] segment_size: u32,
        #[comptime] line_size: u32,
    ) {
        let num_lines_per_segment = segment_size / line_size;

        #[unroll]
        for segment in 0..num_segments {
            #[unroll]
            for line_within_segment in 0..num_lines_per_segment {
                let line = tile.get_line(segment, line_within_segment);
                #[unroll]
                for pos_within_line in 0..line_size {
                    array[segment * segment_size
                        + line_within_segment * line_size
                        + pos_within_line] = line[pos_within_line];
                }
            }
        }
    }

    fn fill_transposed<ES: Numeric>(
        tile: &Tile<ES>,
        array: &mut Array<ES>,
        #[comptime] num_segments: u32,
        #[comptime] segment_size: u32,
        #[comptime] line_size: u32,
    ) {
        let num_lines_per_segment = segment_size / line_size;

        #[unroll]
        for segment in 0..num_segments {
            #[unroll]
            for line_within_segment in 0..num_lines_per_segment {
                let line = tile.get_line(segment, line_within_segment);
                #[unroll]
                for pos_within_line in 0..line_size {
                    array[(line_within_segment * line_size + pos_within_line) * num_segments
                        + segment] = line[pos_within_line];
                }
            }
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
        let m = config.size.m();
        let n = config.size.n();
        let k = config.size.k();

        // 128 a bit arbitrary, but accepts 4x4x4 and rejects 8x8x8
        if m * n * k > 128 {
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
                            "Tile shape in lined axis {:?} should be divisible by line size {:?}",
                            k, lhs
                        )
                    }));
                }
            }
            MatrixLayout::ColMajor => {
                if m % lhs != 0 {
                    return Err(RegisterMatmulConfigError::new(move || {
                        format!(
                            "Tile shape in lined axis {:?} should be divisible by line size {:?}",
                            m, lhs
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
                            "Tile shape in lined axis {:?} should be divisible by line size {:?}",
                            n, rhs
                        )
                    }));
                }
            }
            MatrixLayout::ColMajor => {
                if k % rhs != 0 {
                    return Err(RegisterMatmulConfigError::new(move || {
                        format!(
                            "Tile shape in lined axis {:?} should be divisible by line size {:?}",
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
        line_sizes: &MatmulLineSizes,
        cube_dim: &CubeDim,
        _cube_count: &CubeCount,
        _quantized: bool,
    ) -> Self::Config {
        let (lhs_line_size, rhs_line_size, stage_line_size_update) =
            if input.vectorization.stage_line_size == 0 {
                (line_sizes.lhs as u32, line_sizes.rhs as u32, false)
            } else {
                (
                    input.vectorization.stage_line_size as u32,
                    input.vectorization.stage_line_size as u32,
                    true,
                )
            };
        Config::new(
            input.tile_size,
            cube_dim.x,
            problem.lhs_layout,
            problem.rhs_layout,
            stage_line_size_update,
            lhs_line_size,
            rhs_line_size,
            line_sizes.out as u32,
        )
    }
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for Register instruction
pub struct Config {
    size: TileSize,
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

    fn tile_size(&self) -> &TileSize {
        &self.size
    }
}

impl MatmulConfig for Config {}

impl Config {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        size: TileSize,
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

    fn product_type(&self) -> ProductType {
        // Best algorithm benchmarked on metal
        // Very surprising that RowCol is better in Outer while
        // ColRow is better in Inner
        match (
            self.matrix_layout(Ident::Lhs),
            self.matrix_layout(Ident::Rhs),
        ) {
            (MatrixLayout::RowMajor, MatrixLayout::RowMajor) => ProductType::Inner,
            (MatrixLayout::RowMajor, MatrixLayout::ColMajor) => ProductType::Outer,
            (MatrixLayout::ColMajor, MatrixLayout::RowMajor) => ProductType::Inner,
            (MatrixLayout::ColMajor, MatrixLayout::ColMajor) => ProductType::Outer,
        }
    }
}
