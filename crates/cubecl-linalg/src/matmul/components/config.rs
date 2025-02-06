use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::fmt::{Debug, Display};
use std::hash::Hash;

use crate::matmul::kernels::{matmul::AdvancedConfig, MatmulAvailabilityError};

use super::{MatmulPrecision, MatmulProblem};

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
/// Aggregation of [StageDim]s for all stages
pub struct StageDims {
    pub lhs: LhsStageDim,
    pub rhs: RhsStageDim,
    pub out: OutStageDim,
}

pub trait StageDim: 'static + Send + Sync {
    /// Returns the total number of elements of the stage
    fn total_size(&self) -> u32 {
        self.total_row() * self.total_col()
    }

    /// Returns the number of elements across the x dimension
    fn total_row(&self) -> u32 {
        self.tile_count_row() * self.tile_size_row()
    }

    /// Returns the number of elements across the y dimension
    fn total_col(&self) -> u32 {
        self.tile_count_col() * self.tile_size_col()
    }

    /// Returns the number of elements within one tile
    fn tile_size(&self) -> u32 {
        self.tile_size_row() * self.tile_size_col()
    }

    /// Returns the dimension of a tile across x dimension (rows)
    fn tile_size_row(&self) -> u32;

    /// Returns the dimension of a tile across y dimension (col)
    fn tile_size_col(&self) -> u32;

    fn tile_count(&self) -> u32 {
        self.tile_count_row() * self.tile_count_col()
    }

    /// Returns the number of tiles across x dimension (rows)
    fn tile_count_row(&self) -> u32;

    /// Returns the number of tiles across y dimension (cols)
    fn tile_count_col(&self) -> u32;

    /// Number of elements in a buffer
    fn buffer_num_elements(&self) -> u32;
}

#[derive(CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq)]
/// Dimensions for lhs stage.
pub struct LhsStageDim {
    pub tile_size_row: u32,
    pub tile_size_col: u32,
    pub tile_count_row: u32,
    pub tile_count_col: u32,
}

#[derive(CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq)]
/// Dimensions for rhs stage.
pub struct RhsStageDim {
    pub tile_size_row: u32,
    pub tile_size_col: u32,
    pub tile_count_row: u32,
    pub tile_count_col: u32,
}

#[derive(CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq)]
/// Dimensions for out stage.
pub struct OutStageDim {
    pub tile_size_row: u32,
    pub tile_size_col: u32,
    pub tile_count_row: u32,
    pub tile_count_col: u32,
}

impl StageDim for LhsStageDim {
    fn tile_count_row(&self) -> u32 {
        self.tile_count_row
    }

    fn tile_count_col(&self) -> u32 {
        self.tile_count_col
    }

    fn tile_size_row(&self) -> u32 {
        self.tile_size_row
    }

    fn tile_size_col(&self) -> u32 {
        self.tile_size_col
    }

    fn buffer_num_elements(&self) -> u32 {
        self.tile_count_row * self.tile_size()
    }
}

impl StageDim for RhsStageDim {
    fn tile_count_row(&self) -> u32 {
        self.tile_count_row
    }

    fn tile_count_col(&self) -> u32 {
        self.tile_count_col
    }

    fn tile_size_row(&self) -> u32 {
        self.tile_size_row
    }

    fn tile_size_col(&self) -> u32 {
        self.tile_size_col
    }

    fn buffer_num_elements(&self) -> u32 {
        self.tile_count_col * self.tile_size()
    }
}

impl StageDim for OutStageDim {
    fn tile_count_row(&self) -> u32 {
        self.tile_count_row
    }

    fn tile_count_col(&self) -> u32 {
        self.tile_count_col
    }

    fn tile_size_row(&self) -> u32 {
        self.tile_size_row
    }

    fn tile_size_col(&self) -> u32 {
        self.tile_size_col
    }

    fn buffer_num_elements(&self) -> u32 {
        panic!("Out stage has no concept of buffer")
    }
}
