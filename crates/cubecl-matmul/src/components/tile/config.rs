use crate::components::{MatrixLayout, StageIdent, TileSize};
use std::{fmt::Debug, hash::Hash};

// This serves as interface for higher level matmuls, not for what is used within tile matmul
pub trait TileConfig: Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static {
    /// Returns the number of units in a plane
    fn stage_line_size(&self, ident: StageIdent) -> u32;
    /// Returns the [MatrixLayout] for the given ident
    fn global_line_size(&self, ident: StageIdent) -> u32;
    /// Returns the line size for the given ident
    fn matrix_layout(&self, ident: StageIdent) -> MatrixLayout;
    /// Returns the line size for the given ident
    fn plane_dim(&self) -> u32;
}

/// Configuration for the Tile Matmul level
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SharedTileConfig {
    pub tile_size: TileSize,
    pub plane_dim: u32,
    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,
    pub out_layout: MatrixLayout,
    pub lhs_global_line_size: u32,
    pub rhs_global_line_size: u32,
    pub out_global_line_size: u32,
    pub lhs_stage_line_size: u32,
    pub rhs_stage_line_size: u32,
}

impl TileConfig for SharedTileConfig {
    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }

    fn matrix_layout(&self, ident: StageIdent) -> MatrixLayout {
        match ident {
            StageIdent::Lhs => self.lhs_layout,
            StageIdent::Rhs => self.rhs_layout,
            StageIdent::Acc => MatrixLayout::RowMajor,
            StageIdent::Out => MatrixLayout::RowMajor,
        }
    }

    fn stage_line_size(&self, ident: StageIdent) -> u32 {
        match ident {
            StageIdent::Lhs => self.lhs_stage_line_size,
            StageIdent::Rhs => self.rhs_stage_line_size,
            StageIdent::Acc => self.out_global_line_size,
            StageIdent::Out => self.out_global_line_size,
        }
    }

    fn global_line_size(&self, ident: StageIdent) -> u32 {
        match ident {
            StageIdent::Lhs => self.lhs_global_line_size,
            StageIdent::Rhs => self.rhs_global_line_size,
            StageIdent::Acc => self.out_global_line_size,
            StageIdent::Out => self.out_global_line_size,
        }
    }
}

impl SharedTileConfig {
    pub fn new(
        tile_size: TileSize,
        plane_dim: u32,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        lhs_global_line_size: u32,
        rhs_global_line_size: u32,
        out_global_line_size: u32,
        lhs_stage_line_size: u32,
        rhs_stage_line_size: u32,
    ) -> Self {
        SharedTileConfig {
            tile_size,
            plane_dim,
            lhs_layout,
            rhs_layout,
            out_layout: MatrixLayout::RowMajor,
            lhs_global_line_size,
            rhs_global_line_size,
            out_global_line_size,
            lhs_stage_line_size,
            rhs_stage_line_size,
        }
    }
}
