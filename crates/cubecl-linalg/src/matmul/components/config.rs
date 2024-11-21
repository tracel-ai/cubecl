use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::fmt::Debug;
use std::hash::Hash;

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
            Ident::Out => panic!(),
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
    pub lhs: StageDim,
    pub rhs: StageDim,
    pub out: StageDim,
}

#[derive(CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq)]
/// Dimensions for a stage. A stage has `num_tiles_x` tiles of size `tile_size_x` in
/// x direction, and `num_tiles_y` tiles of size `tile_size_y` in y dimension.
///
/// Dimensions x and y are respectively the row and column dimensions,
/// regardless of the [MatrixLayout]:
///  - Lhs: x=m, y=k
///  - Rhs: x=k, y=n
///  - Out: x=m, y=n
pub struct StageDim {
    pub tile_size_x: u32,
    pub tile_size_y: u32,
    pub num_tiles_x: u32,
    pub num_tiles_y: u32,
    pub num_tiles_per_buffer: u32,
}

impl StageDim {
    pub fn new(
        ident: Ident,
        tile_size_x: u32,
        tile_size_y: u32,
        num_tiles_x: u32,
        num_tiles_y: u32,
    ) -> Self {
        Self {
            tile_size_x,
            tile_size_y,
            num_tiles_x,
            num_tiles_y,
            num_tiles_per_buffer: match ident {
                Ident::Lhs => num_tiles_x,
                Ident::Rhs => num_tiles_y,
                Ident::Out => 0,
            },
        }
    }

    /// Returns the total number of elements of the stage
    pub fn num_elements(&self) -> u32 {
        self.num_tiles_x * self.num_tiles_y * self.tile_num_elements()
    }

    /// Returns the number of elements within one tile
    pub fn tile_num_elements(&self) -> u32 {
        self.tile_size_x * self.tile_size_y
    }

    /// Returns the height of the stage, i.e. the number of elements across the x dimension
    pub fn num_elements_x_dim(&self) -> u32 {
        self.num_tiles_x * self.tile_size_x
    }

    /// Returns the width of the stage, i.e. the number of elements across the y dimension
    pub fn num_elements_y_dim(&self) -> u32 {
        self.num_tiles_y * self.tile_size_y
    }

    pub fn buffer_num_elements(&self) -> u32 {
        // Would be cleaner with an option, but it's not supported for CubeType
        if self.num_tiles_per_buffer == 0 {
            panic!("Should not call buffer_num_elements on output")
        }
        self.tile_num_elements() * self.num_tiles_per_buffer
    }
}
