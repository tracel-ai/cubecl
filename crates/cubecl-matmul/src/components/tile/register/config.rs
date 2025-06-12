use crate::components::config::MatmulConfig;
use crate::components::tile::TileConfig;
use crate::components::{Ident, MatrixLayout, TileSize};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

pub enum ProductType {
    /// Needs lhs to be row major and rhs to be col major
    /// If not the case, tile will be transposed
    Inner,
    /// Needs lhs to be col major and rhs to be row major
    /// If not the case, tile will be transposed
    Outer,
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for Register instruction
pub struct RegisterConfig {
    size: TileSize,
    plane_dim: u32,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    pub stage_dynamic_line_size: bool,
    lhs_line_size: u32,
    rhs_line_size: u32,
    out_line_size: u32,
}

impl TileConfig for RegisterConfig {
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

impl MatmulConfig for RegisterConfig {}

impl RegisterConfig {
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

    pub fn product_type(&self) -> ProductType {
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
