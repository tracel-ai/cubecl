use crate::matmul::config::{ComptimeConfig, MatmulConfig};
use crate::matmul::matmul_tile::TmmConfig;
use crate::matmul::matrix::{Ident, MatrixLayout};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct CmmaTileMatmulConfig {
    plane_dim: u32,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
}

impl ComptimeConfig for CmmaTileMatmulConfig {}

impl TmmConfig for CmmaTileMatmulConfig {
    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }

    fn layout(&self, ident: Ident) -> MatrixLayout {
        match ident {
            Ident::Lhs => self.lhs_layout,
            Ident::Rhs => self.rhs_layout,
            Ident::Out => MatrixLayout::RowMajor,
        }
    }
}

impl MatmulConfig for CmmaTileMatmulConfig {}

impl CmmaTileMatmulConfig {
    pub fn new(plane_dim: u32, lhs_layout: MatrixLayout, rhs_layout: MatrixLayout) -> Self {
        Self {
            plane_dim,
            lhs_layout,
            rhs_layout,
        }
    }
}
