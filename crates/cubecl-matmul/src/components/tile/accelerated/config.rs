use crate::components::config::MatmulConfig;
use crate::components::tile::TileConfig;
use crate::components::{Ident, MatrixLayout, TileSize};
use cubecl_core::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for Accelerated instruction
pub struct AcceleratedConfig {
    size: TileSize,
    plane_dim: u32,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    pub stage_dynamic_line_size: bool,
    lhs_line_size: u32,
    rhs_line_size: u32,
    out_line_size: u32,
}

impl TileConfig for AcceleratedConfig {
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

impl MatmulConfig for AcceleratedConfig {}

impl AcceleratedConfig {
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
}
