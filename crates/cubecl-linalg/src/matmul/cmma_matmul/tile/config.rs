use crate::matmul::config::{ComptimeConfig, MatmulConfig};
use crate::matmul::matmul_tile::TmmConfig;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct CmmaTileMatmulConfig {
    plane_dim: u32,
}

impl ComptimeConfig for CmmaTileMatmulConfig {}

impl TmmConfig for CmmaTileMatmulConfig {
    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }
}

impl MatmulConfig for CmmaTileMatmulConfig {}

impl CmmaTileMatmulConfig {
    pub fn new(plane_dim: u32) -> Self {
        Self { plane_dim }
    }
}
