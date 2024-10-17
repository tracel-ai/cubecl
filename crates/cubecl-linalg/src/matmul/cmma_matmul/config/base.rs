use cubecl_core::prelude::*;

use crate::matmul::{
    config::{ConfigBuilder, MatmulConfig},
    matmul_global::GmmConfig,
    matmul_stage::SmmConfig,
    matmul_tile::TmmConfig,
    matrix_layout::MatrixLayout,
    problem::MatmulProblem,
};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct CmmaConfig {
    pub out_smem_line_size: u32,
    pub layouts: (MatrixLayout, MatrixLayout),
    pub num_planes: u32,
    pub plane_dim: u32,
}

#[derive(Default)]
pub struct CmmaConfigBuilder {
    pub plane_dim: Option<u32>,
    pub num_planes: Option<u32>,
}

impl ConfigBuilder for CmmaConfigBuilder {
    type Config = CmmaConfig;

    fn from_cube_settings(&self, cube_dim: &CubeDim, _cube_count: &CubeCount) -> Self {
        Self {
            plane_dim: Some(cube_dim.x),
            num_planes: Some(cube_dim.y),
        }
    }

    fn from_problem(&self, problem: &MatmulProblem) -> Self::Config {
        Self::Config {
            out_smem_line_size: problem.out_line_size as u32,
            layouts: (problem.lhs_layout, problem.rhs_layout),
            num_planes: self.num_planes.unwrap(),
            plane_dim: self.plane_dim.unwrap(),
        }
    }
}

impl MatmulConfig for CmmaConfig {
    type ConfigBuilder = CmmaConfigBuilder;

    fn build() -> Self::ConfigBuilder {
        Self::ConfigBuilder::default()
    }

    fn num_planes(&self) -> u32 {
        self.num_planes
    }

    fn plane_dim(&self) -> u32 {
        self.plane_dim
    }
}

impl GmmConfig for CmmaConfig {
    type SmmConfig = Self;

    fn into_smm_config(self) -> Self::SmmConfig {
        self
    }
}

impl SmmConfig for CmmaConfig {
    type TmmConfig = Self;

    fn into_tmm_config(self) -> Self::TmmConfig {
        self
    }
}

impl TmmConfig for CmmaConfig {}

impl Init for CmmaConfig {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

impl CubeType for CmmaConfig {
    type ExpandType = Self;
}
