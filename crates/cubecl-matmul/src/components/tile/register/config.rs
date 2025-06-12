use crate::components::config::MatmulConfig;
use crate::components::tile::TileConfig;
use crate::components::{Ident, MatrixLayout, TileSize, TilingScheme};

pub enum ProductType {
    /// Needs lhs to be row major and rhs to be col major
    /// If not the case, tile will be transposed
    Inner,
    /// Needs lhs to be col major and rhs to be row major
    /// If not the case, tile will be transposed
    Outer,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for Register instruction
pub struct RegisterConfig {
    tiling_scheme: TilingScheme,
    plane_dim: u32,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    pub stage_dynamic_line_size: bool,
    lhs_global_line_size: u32,
    rhs_global_line_size: u32,
    out_global_line_size: u32,
    lhs_stage_line_size: u32,
    rhs_stage_line_size: u32,
}

impl TileConfig for RegisterConfig {
    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }

    fn matrix_layout<I: Into<Ident>>(&self, ident: I) -> MatrixLayout {
        match ident.into() {
            Ident::Lhs => self.lhs_layout,
            Ident::Rhs => self.rhs_layout,
            Ident::Out => MatrixLayout::RowMajor,
        }
    }

    fn stage_line_size<I: Into<Ident>>(&self, ident: I) -> u32 {
        match ident.into() {
            Ident::Lhs => self.lhs_stage_line_size,
            Ident::Rhs => self.rhs_stage_line_size,
            Ident::Out => self.out_global_line_size,
        }
    }

    fn global_line_size<I: Into<Ident>>(&self, ident: I) -> u32 {
        match ident.into() {
            Ident::Lhs => self.lhs_global_line_size,
            Ident::Rhs => self.rhs_global_line_size,
            Ident::Out => self.out_global_line_size,
        }
    }

    fn tile_size(&self) -> &TileSize {
        &self.tiling_scheme.tile_size
    }
}

impl MatmulConfig for RegisterConfig {}

impl RegisterConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tiling_scheme: TilingScheme,
        plane_dim: u32,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        stage_dynamic_line_size: bool,
        lhs_global_line_size: u32,
        rhs_global_line_size: u32,
        out_global_line_size: u32,
        lhs_stage_line_size: u32,
        rhs_stage_line_size: u32,
    ) -> Self {
        Self {
            tiling_scheme,
            plane_dim,
            lhs_layout,
            rhs_layout,
            stage_dynamic_line_size,
            lhs_global_line_size,
            rhs_global_line_size,
            out_global_line_size,
            lhs_stage_line_size,
            rhs_stage_line_size,
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
