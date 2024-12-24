use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::fmt::{Debug, Display};
use std::hash::Hash;

use crate::matmul::kernels::{matmul::AdvancedConfig, MatmulAvailabilityError};

use super::{MatmulPrecision, MatmulProblem};

pub type InvalidConfigError = Box<dyn Display>;

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
    CubeType + Copy + Clone + Send + Sync + 'static + Eq + PartialEq + Hash + Debug + IntoRuntime
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
    fn total_elements(&self) -> u32 {
        self.num_elements_x_dim() * self.num_elements_y_dim()
    }

    /// Returns the number of elements within one tile
    fn tile_num_elements(&self) -> u32 {
        self.tile_size_x_dim() * self.tile_size_y_dim()
    }

    /// Returns the number of elements across the x dimension
    fn num_elements_x_dim(&self) -> u32 {
        self.num_tiles_x_dim() * self.tile_size_x_dim()
    }

    /// Returns the number of elements across the y dimension
    fn num_elements_y_dim(&self) -> u32 {
        self.num_tiles_y_dim() * self.tile_size_y_dim()
    }

    fn num_tiles(&self) -> u32 {
        self.num_tiles_x_dim() * self.num_tiles_y_dim()
    }

    /// Number of elements in a buffer
    fn buffer_num_elements(&self) -> u32;

    /// Returns the number of tiles across x dimension (rows)
    fn num_tiles_x_dim(&self) -> u32;

    /// Returns the number of tiles across y dimension (cols)
    fn num_tiles_y_dim(&self) -> u32;

    /// Returns the dimension of a tile across x dimension (rows)
    fn tile_size_x_dim(&self) -> u32;

    /// Returns the dimension of a tile across y dimension (col)
    fn tile_size_y_dim(&self) -> u32;
}

#[derive(CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq)]
/// Dimensions for lhs stage.
pub struct LhsStageDim {
    pub tile_size_m: u32,
    pub tile_size_k: u32,
    pub num_tiles_m: u32,
    pub num_tiles_k: u32,
}

#[derive(CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq)]
/// Dimensions for rhs stage.
pub struct RhsStageDim {
    pub tile_size_k: u32,
    pub tile_size_n: u32,
    pub num_tiles_k: u32,
    pub num_tiles_n: u32,
}

#[derive(CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq)]
/// Dimensions for out stage.
pub struct OutStageDim {
    pub tile_size_m: u32,
    pub tile_size_n: u32,
    pub num_tiles_m: u32,
    pub num_tiles_n: u32,
}

impl StageDim for LhsStageDim {
    fn num_tiles_x_dim(&self) -> u32 {
        self.num_tiles_m
    }

    fn num_tiles_y_dim(&self) -> u32 {
        self.num_tiles_k
    }

    fn tile_size_x_dim(&self) -> u32 {
        self.tile_size_m
    }

    fn tile_size_y_dim(&self) -> u32 {
        self.tile_size_k
    }

    fn buffer_num_elements(&self) -> u32 {
        self.num_tiles_m * self.tile_num_elements()
    }
}

impl StageDim for RhsStageDim {
    fn num_tiles_x_dim(&self) -> u32 {
        self.num_tiles_k
    }

    fn num_tiles_y_dim(&self) -> u32 {
        self.num_tiles_n
    }

    fn tile_size_x_dim(&self) -> u32 {
        self.tile_size_k
    }

    fn tile_size_y_dim(&self) -> u32 {
        self.tile_size_n
    }

    fn buffer_num_elements(&self) -> u32 {
        self.num_tiles_n * self.tile_num_elements()
    }
}

impl StageDim for OutStageDim {
    fn num_tiles_x_dim(&self) -> u32 {
        self.num_tiles_m
    }

    fn num_tiles_y_dim(&self) -> u32 {
        self.num_tiles_n
    }

    fn tile_size_x_dim(&self) -> u32 {
        self.tile_size_m
    }

    fn tile_size_y_dim(&self) -> u32 {
        self.tile_size_n
    }

    fn buffer_num_elements(&self) -> u32 {
        panic!("Out stage has no concept of buffer")
    }
}
