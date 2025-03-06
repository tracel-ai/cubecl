use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::fmt::{Debug, Display};
use std::hash::Hash;

use crate::matmul::kernels::{matmul::AdvancedConfig, MatmulAvailabilityError};

use super::{MatmulPrecision, MatmulProblem, MatmulSize};

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
    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        _client: &ComputeClient<R::Server, R::Channel>,
        _config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError>;

    /// Create config for this matmul, given launch information
    fn make_config(
        input: Self::Input,
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
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
    pub fn as_input(&self) -> InputIdent {
        match self {
            Ident::Lhs => InputIdent::Lhs,
            Ident::Rhs => InputIdent::Rhs,
            Ident::Out => panic!("Out is not an input."),
        }
    }
}

pub enum InputIdent {
    Lhs,
    Rhs,
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

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
/// Aggregation of [StageTiling]s for all components.
pub struct CompleteStageTiling {
    pub tile_shape: MatmulSize,
    pub tile_count: MatmulSize,
}

impl CompleteStageTiling {
    pub fn get(&self, ident: Ident) -> TilingDimensions {
        TilingDimensions::Full(match ident {
            Ident::Lhs => TilingDimensionsInner {
                tile_shape_row: self.tile_shape.m,
                tile_shape_col: self.tile_shape.k,
                tile_count_row: self.tile_count.m,
                tile_count_col: self.tile_count.k,
            },
            Ident::Rhs => TilingDimensionsInner {
                tile_shape_row: self.tile_shape.k,
                tile_shape_col: self.tile_shape.n,
                tile_count_row: self.tile_count.k,
                tile_count_col: self.tile_count.n,
            },
            Ident::Out => TilingDimensionsInner {
                tile_shape_row: self.tile_shape.m,
                tile_shape_col: self.tile_shape.n,
                tile_count_row: self.tile_count.m,
                tile_count_col: self.tile_count.n,
            },
        })
    }

    pub fn total_shape(&self) -> MatmulSize {
        MatmulSize {
            m: self.tile_shape.m * self.tile_count.m,
            n: self.tile_shape.n * self.tile_count.n,
            k: self.tile_shape.k * self.tile_count.k,
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
/// Dimensions for stage.
pub struct TilingDimensionsInner {
    pub tile_shape_row: u32,
    pub tile_shape_col: u32,
    pub tile_count_row: u32,
    pub tile_count_col: u32,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
/// Dimensions for stage.
pub enum TilingDimensions {
    Full(TilingDimensionsInner),
}

impl TilingDimensions {
    /// Returns the total number of elements of the stage.
    pub fn total_size(&self) -> u32 {
        match self {
            TilingDimensions::Full(inner) => inner.total_size(),
        }
    }

    /// Returns the total number of rows of the stage.
    pub fn total_row(&self) -> u32 {
        match self {
            TilingDimensions::Full(inner) => inner.total_row(),
        }
    }

    /// Returns the total number of columns of the stage.
    pub fn total_col(&self) -> u32 {
        match self {
            TilingDimensions::Full(inner) => inner.total_col(),
        }
    }

    /// Returns the number of elements within one tile.
    pub fn tile_size(&self) -> u32 {
        match self {
            TilingDimensions::Full(inner) => inner.tile_size(),
        }
    }

    /// Returns the size of the row axis of a tile.
    pub fn tile_shape_row(&self) -> u32 {
        match self {
            TilingDimensions::Full(inner) => inner.tile_shape_row(),
        }
    }

    /// Returns the size of the column axis of a tile.
    pub fn tile_shape_col(&self) -> u32 {
        match self {
            TilingDimensions::Full(inner) => inner.tile_shape_col(),
        }
    }

    /// Returns the number of tiles within the stage.
    pub fn tile_count(&self) -> u32 {
        match self {
            TilingDimensions::Full(inner) => inner.tile_count(),
        }
    }

    /// Returns the number of tiles across the row axis of the stage.
    pub fn tile_count_row(&self) -> u32 {
        match self {
            TilingDimensions::Full(inner) => inner.tile_count_row(),
        }
    }

    /// Returns the number of tiles across the column axis of the stage.
    pub fn tile_count_col(&self) -> u32 {
        match self {
            TilingDimensions::Full(inner) => inner.tile_count_col(),
        }
    }
}

impl TilingDimensionsInner {
    /// Returns the total number of elements of the stage.
    pub fn total_size(&self) -> u32 {
        self.total_row() * self.total_col()
    }

    /// Returns the total number of rows of the stage.
    pub fn total_row(&self) -> u32 {
        self.tile_count_row() * self.tile_shape_row()
    }

    /// Returns the total number of columns of the stage.
    pub fn total_col(&self) -> u32 {
        self.tile_count_col() * self.tile_shape_col()
    }

    /// Returns the number of elements within one tile.
    pub fn tile_size(&self) -> u32 {
        self.tile_shape_row() * self.tile_shape_col()
    }

    /// Returns the size of the row axis of a tile.
    pub fn tile_shape_row(&self) -> u32 {
        self.tile_shape_row
    }

    /// Returns the size of the column axis of a tile.
    pub fn tile_shape_col(&self) -> u32 {
        self.tile_shape_col
    }

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
