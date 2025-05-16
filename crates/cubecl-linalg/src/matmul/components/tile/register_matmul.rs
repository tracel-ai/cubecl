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

pub enum ProductType {
    /// Needs lhs to be row major and rhs to be col major
    /// If not the case, tile will be preloaded and transposed
    Inner,
    /// Needs lhs to be col major and rhs to be row major
    /// If not the case, tile will be preloaded and transposed
    Outer,
}

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
    data: Array<EA>,
    #[cube(comptime)]
    rows: u32,
    #[cube(comptime)]
    cols: u32,
}

#[derive(CubeType)]
pub enum TileInput<ES: Numeric> {
    OnTheFly(ComptimeCell<CubeOption<Tile<ES>>>),
    Register(Array<ES>),
}

#[derive(CubeType, Copy, Clone)]
// A row or column, the length of a line
pub enum Segment<ES: Numeric> {
    Line(Line<ES>),
    Offset(u32),
}

#[cube]
impl<ES: Numeric> TileInput<ES> {
    fn prepare_segment(&self, segment_index: u32, length: u32) -> Segment<ES> {
        match self {
            TileInput::OnTheFly(comptime_cell) => {
                let line = comptime_cell
                    .read()
                    .unwrap()
                    .get_segment_as_one_line(segment_index);
                Segment::new_Line(line)
            }
            TileInput::Register(_) => Segment::new_Offset(length),
        }
    }

    fn get_elem_in_segment(&self, segment: Segment<ES>, element_index: u32) -> ES {
        match self {
            TileInput::OnTheFly(_) => match segment {
                Segment::Line(line) => line[element_index],
                Segment::Offset(_) => comptime!(unreachable!()),
            },
            TileInput::Register(array) => match segment {
                Segment::Line(_) => comptime!(unreachable!()),
                Segment::Offset(offset) => array[offset + element_index],
            },
        }
    }
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
        acc: &mut Self::Accumulator,
        #[comptime] config: Config,
    ) {
        match config.product_type() {
            ProductType::Inner => Self::inner_product(lhs, rhs, acc, config),
            ProductType::Outer => Self::outer_product(lhs, rhs, acc, config),
        }
    }

    fn allocate_lhs(#[comptime] config: Config) -> Self::Lhs {
        match (config.product_type(), config.lhs_layout) {
            (ProductType::Inner, MatrixLayout::RowMajor)
            | (ProductType::Outer, MatrixLayout::ColMajor) => {
                TileInput::new_OnTheFly(ComptimeCell::new(CubeOption::new_None()))
            }
            (ProductType::Inner, MatrixLayout::ColMajor)
            | (ProductType::Outer, MatrixLayout::RowMajor) => {
                TileInput::new_Register(Array::new(config.size.m * config.size.k))
            }
        }
    }

    fn allocate_rhs(#[comptime] config: Config) -> Self::Rhs {
        match (config.product_type(), config.rhs_layout) {
            (ProductType::Inner, MatrixLayout::ColMajor)
            | (ProductType::Outer, MatrixLayout::RowMajor) => {
                TileInput::new_OnTheFly(ComptimeCell::new(CubeOption::new_None()))
            }
            (ProductType::Inner, MatrixLayout::RowMajor)
            | (ProductType::Outer, MatrixLayout::ColMajor) => {
                TileInput::new_Register(Array::new(config.size.k * config.size.n))
            }
        }
    }

    fn fill_lhs(tile: &Tile<MP::ES>, lhs: &mut Self::Lhs, #[comptime] config: Config) {
        match lhs {
            TileInput::OnTheFly(comptime_cell) => {
                comptime_cell.store(CubeOption::new_Some(comptime!(tile.clone())))
            }
            TileInput::Register(array) => match config.lhs_layout {
                MatrixLayout::RowMajor => {
                    assert!(config.lhs_line_size == config.size.k);
                    Self::fill_transposed(tile, array, config.size.m, config.size.k);
                }
                MatrixLayout::ColMajor => {
                    assert!(config.lhs_line_size == config.size.m);
                    Self::fill_transposed(tile, array, config.size.k, config.size.m);
                }
            },
        }
    }

    fn fill_rhs(tile: &Tile<MP::ES>, rhs: &mut Self::Rhs, #[comptime] config: Config) {
        match rhs {
            TileInput::OnTheFly(comptime_cell) => {
                comptime_cell.store(CubeOption::new_Some(comptime!(tile.clone())))
            }
            TileInput::Register(array) => match config.rhs_layout {
                MatrixLayout::RowMajor => {
                    assert!(config.rhs_line_size == config.size.n);
                    Self::fill_transposed(tile, array, config.size.k, config.size.n);
                }
                MatrixLayout::ColMajor => {
                    assert!(config.rhs_line_size == config.size.k);
                    Self::fill_transposed(tile, array, config.size.n, config.size.k);
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
        let rows = config.size.m;
        let cols = config.size.n;

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
        lhs: &TileInput<ES>,
        rhs: &TileInput<ES>,
        acc: &mut TileAccumulator<EA>,
        #[comptime] config: Config,
    ) {
        let (m, n, k) = comptime! {let (m, n, k) = config.size.into(); (m, n, k)};

        #[unroll]
        for m_ in 0..m {
            let lhs_segment = lhs.prepare_segment(m_, m_ * k);
            #[unroll]
            for n_ in 0..n {
                let rhs_segment = rhs.prepare_segment(n_, n_ * k);
                #[unroll]
                for k_ in 0..k {
                    let lhs_elem = EA::cast_from(lhs.get_elem_in_segment(lhs_segment, k_));
                    let rhs_elem = EA::cast_from(rhs.get_elem_in_segment(rhs_segment, k_));
                    acc.data[m_ * n + n_] += lhs_elem * rhs_elem;
                }
            }
        }
    }

    fn outer_product<ES: Numeric, EA: Numeric>(
        lhs: &TileInput<ES>,
        rhs: &TileInput<ES>,
        acc: &mut TileAccumulator<EA>,
        #[comptime] config: Config,
    ) {
        let (m, n, k) = comptime! {let (m, n, k) = config.size.into(); (m, n, k)};

        #[unroll]
        for k_ in 0..k {
            let lhs_segment = lhs.prepare_segment(k_, k_ * m);
            let rhs_segment = rhs.prepare_segment(k_, k_ * n);
            #[unroll]
            for m_ in 0..m {
                let lhs_elem = EA::cast_from(lhs.get_elem_in_segment(lhs_segment, m_));
                #[unroll]
                for n_ in 0..n {
                    let rhs_elem = EA::cast_from(rhs.get_elem_in_segment(rhs_segment, n_));
                    acc.data[m_ * n + n_] += lhs_elem * rhs_elem;
                }
            }
        }
    }

    fn fill_transposed<ES: Numeric>(
        tile: &Tile<ES>,
        array: &mut Array<ES>,
        num_lines: u32,
        line_length: u32,
    ) {
        #[unroll]
        for line_index in 0..num_lines {
            let line = tile.get_segment_as_one_line(line_index);
            #[unroll]
            for pos_within_line in 0..line_length {
                array[pos_within_line * line_length + line_index] = line[pos_within_line];
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
        let m = config.size.m;
        let n = config.size.n;
        let k = config.size.k;

        // TODO this is selector logic
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

        if k != m || m != lhs || lhs != rhs || rhs != n {
            return Err(RegisterMatmulConfigError::new(move || {
                format!(
                    "Input line and tile sizes must all match. Got m={m:?}, n={n:?}, k={k:?}, lhs={lhs:?}, rhs={rhs:?}"
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

    fn product_type(&self) -> ProductType {
        match (
            self.matrix_layout(Ident::Lhs),
            self.matrix_layout(Ident::Rhs),
        ) {
            // We should benchmark, but normally:
            // Col-Row should be Outer (probably the best case)
            // Row-Col should be Inner
            // Row-Row and Col-Col are unclear
            (MatrixLayout::RowMajor, MatrixLayout::RowMajor) => ProductType::Inner,
            (MatrixLayout::RowMajor, MatrixLayout::ColMajor) => ProductType::Inner,
            (MatrixLayout::ColMajor, MatrixLayout::RowMajor) => ProductType::Inner,
            (MatrixLayout::ColMajor, MatrixLayout::ColMajor) => ProductType::Inner,
        }
    }
}
