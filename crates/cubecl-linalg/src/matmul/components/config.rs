use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::fmt::{Debug, Display};
use std::hash::Hash;

use crate::matmul::kernels::MatmulAvailabilityError;

use super::problem::MatmulLineSizes;
use super::{MatmulPrecision, MatmulProblem, TilingScheme};

pub type InvalidConfigError = Box<dyn Display>;

pub struct FormattedConfigError {
    func: Box<dyn Fn() -> String>,
}

impl FormattedConfigError {
    #[allow(clippy::new_ret_no_self)]
    pub fn new<F: Fn() -> String + 'static>(func: F) -> Box<dyn Display> {
        Box::new(Self {
            func: Box::new(func),
        })
    }
}

impl Display for FormattedConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = (self.func)();
        write!(f, "{string}")
    }
}

/// Provides configuration for a matmul kernel at any level
pub trait MatmulConfigFactory: Send + Sync + 'static {
    /// Configuration tailored to the matmul implementation
    type Config: MatmulConfig;
    type Input;

    /// Asserts that the configuration for this matmul will lead to a valid computation
    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError>;

    /// Checks if the client can handle the features used in this computation
    #[allow(clippy::result_large_err)]
    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        _client: &ComputeClient<R::Server, R::Channel>,
        _config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError>;

    /// Create config for this matmul, given launch information
    fn make_config(
        input: Self::Input,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        quantized: bool,
    ) -> Self::Config;
}

/// A config for a matmul
///
/// Useful to aggregate many trait bounds
pub trait MatmulConfig:
    Copy + Clone + Send + Sync + 'static + Eq + PartialEq + Hash + Debug
{
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
/// Identifier for all three tensors in a matmul
///
/// Useful to specialize some functions depending on the tensor
pub enum Ident {
    Lhs,
    Rhs,
    Out,
}

impl Ident {
    pub fn as_input_ident(&self) -> InputIdent {
        match self {
            Ident::Lhs => InputIdent::Lhs,
            Ident::Rhs => InputIdent::Rhs,
            Ident::Out => panic!("Out is not an input."),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
/// Identifier for the two input tensors in a matmul.
///
/// Useful to specialize some functions depending on the tensor
pub enum InputIdent {
    Lhs,
    Rhs,
}

impl InputIdent {
    pub fn as_ident(&self) -> Ident {
        match self {
            InputIdent::Lhs => Ident::Lhs,
            InputIdent::Rhs => Ident::Rhs,
        }
    }
}

impl From<InputIdent> for Ident {
    fn from(value: InputIdent) -> Self {
        value.as_ident()
    }
}

#[derive(CubeType, Copy, Clone, PartialEq, Eq, Hash, Debug)]
/// Layout of a 2D structure such as a tensor, shared memory or slice,
/// used within any matmul kernel level
pub enum MatrixLayout {
    RowMajor,
    ColMajor,
}

#[cube]
/// Maps the matmul MatrixLayout to cmma's MatrixLayout, for use in Cmma API.
pub fn as_cmma_layout(#[comptime] layout: MatrixLayout) -> cmma::MatrixLayout {
    match layout {
        MatrixLayout::RowMajor => cmma::MatrixLayout::RowMajor,
        MatrixLayout::ColMajor => cmma::MatrixLayout::ColMajor,
    }
}

impl TilingDimensions {
    pub fn new(tiling_scheme: &TilingScheme, ident: Ident) -> TilingDimensions {
        match ident {
            Ident::Lhs => TilingDimensions {
                tile_size_row: tiling_scheme.elements_in_tile_m(),
                tile_size_col: tiling_scheme.elements_in_tile_k(),
                tile_count_row: tiling_scheme.tiles_in_stage_m(),
                tile_count_col: tiling_scheme.tiles_in_stage_k(),
            },
            Ident::Rhs => TilingDimensions {
                tile_size_row: tiling_scheme.elements_in_tile_k(),
                tile_size_col: tiling_scheme.elements_in_tile_n(),
                tile_count_row: tiling_scheme.tiles_in_stage_k(),
                tile_count_col: tiling_scheme.tiles_in_stage_n(),
            },
            Ident::Out => TilingDimensions {
                tile_size_row: tiling_scheme.elements_in_tile_m(),
                tile_size_col: tiling_scheme.elements_in_tile_n(),
                tile_count_row: tiling_scheme.tiles_in_stage_m(),
                tile_count_col: tiling_scheme.tiles_in_stage_n(),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
/// Dimensions for stage.
pub struct TilingDimensions {
    pub tile_size_row: u32,
    pub tile_size_col: u32,
    pub tile_count_row: u32,
    pub tile_count_col: u32,
}

impl TilingDimensions {
    /// Returns the total number of elements of the stage.
    // pub fn total_size(&self) -> u32 {
    //     self.total_row() * self.total_col()
    // }

    /// Returns the total number of rows of the stage.
    // pub fn total_row(&self) -> u32 {
    //     self.tile_count_row() * self.tile_size_row()
    // }

    /// Returns the total number of columns of the stage.
    // pub fn total_col(&self) -> u32 {
    //     self.tile_count_col() * self.tile_size_col()
    // }

    /// Returns the number of elements within one tile.
    // pub fn tile_size(&self) -> u32 {
    //     self.tile_size_row() * self.tile_size_col()
    // }

    /// Returns the size of the row axis of a tile.
    // pub fn tile_size_row(&self) -> u32 {
    //     self.tile_size_row
    // }

    /// Returns the size of the column axis of a tile.
    // pub fn tile_size_col(&self) -> u32 {
    //     self.tile_size_col
    // }

    /// Returns the number of tiles within the stage.
    pub fn tile_count(&self) -> u32 {
        self.tile_count_row() * self.tile_count_col()
    }

    /// Returns the number of tiles across the row axis of the stage.
    pub fn tile_count_row(&self) -> u32 {
        self.tile_count_row
    }

    /// Returns the number of tiles across the column axis of the stage.
    pub fn tile_count_col(&self) -> u32 {
        self.tile_count_col
    }
}

pub trait TensorIdent:
    Clone + Copy + Debug + Hash + PartialEq + Eq + Send + Sync + 'static
{
    const IDENT: Ident;
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Lhs;
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Rhs;
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Out;

impl TensorIdent for Lhs {
    const IDENT: Ident = Ident::Lhs;
}

impl TensorIdent for Rhs {
    const IDENT: Ident = Ident::Rhs;
}

impl TensorIdent for Out {
    const IDENT: Ident = Ident::Out;
}
