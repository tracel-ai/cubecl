use cubecl_core::prelude::*;

use crate::matmul::{
    config::{ConfigBuilder, MatmulConfig, MatmulPreConfig},
    matmul_global::GmmConfig,
    matmul_stage::SmmConfig,
    matmul_tile::TmmConfig,
    matrix::{Ident, MatrixLayout},
    problem::MatmulProblem,
};

use super::{StageDim, StageDims};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct CmmaConfig {
    pub out_smem_line_size: u32,
    pub layouts: (MatrixLayout, MatrixLayout),
    pub num_planes: u32,
    pub plane_dim: u32,
    pub stage_dims: StageDims,
    pub line_sizes: (u32, u32, u32),
}

#[derive(Default)]
pub struct CmmaConfigBuilder {
    pub plane_dim: Option<u32>,
    pub num_planes: Option<u32>,
    pub out_smem_line_size: Option<u32>,
    pub layouts: Option<(MatrixLayout, MatrixLayout)>,
    pub stage_dims: Option<StageDims>,
    pub line_sizes: Option<(u32, u32, u32)>,
}

impl ConfigBuilder for CmmaConfigBuilder {
    type Config = CmmaConfig;

    fn from_cube_settings(mut self, cube_dim: &CubeDim, _cube_count: &CubeCount) -> Self {
        self.plane_dim = Some(cube_dim.x);
        self.num_planes = Some(cube_dim.y);
        self
    }

    fn from_problem(mut self, problem: &MatmulProblem) -> Self {
        self.out_smem_line_size = Some(problem.out_line_size as u32);
        self.layouts = Some((problem.lhs_layout, problem.rhs_layout));
        self.line_sizes = Some((
            problem.lhs_line_size as u32,
            problem.rhs_line_size as u32,
            problem.out_line_size as u32,
        ));
        self
    }

    fn into_config(self) -> Self::Config {
        Self::Config {
            out_smem_line_size: self.out_smem_line_size.unwrap(),
            layouts: self.layouts.unwrap(),
            num_planes: self.num_planes.unwrap(),
            plane_dim: self.plane_dim.unwrap(),
            stage_dims: self.stage_dims.unwrap(),
            line_sizes: self.line_sizes.unwrap(),
        }
    }
}

#[derive(Default)]
pub struct CmmaPreConfig {
    pub lhs_tile_size_x: u32,
    pub lhs_tile_size_y: u32,

    pub rhs_tile_size_x: u32,
    pub rhs_tile_size_y: u32,

    pub out_tile_size_x: u32,
    pub out_tile_size_y: u32,

    pub lhs_num_tiles_x: Option<u32>,
    pub lhs_num_tiles_y: Option<u32>,

    pub rhs_num_tiles_x: Option<u32>,
    pub rhs_num_tiles_y: Option<u32>,

    pub out_num_tiles_x: Option<u32>,
    pub out_num_tiles_y: Option<u32>,
}

impl MatmulPreConfig for CmmaPreConfig {
    type MatmulConfig = CmmaConfig;
    type ConfigBuilder = CmmaConfigBuilder;

    fn into_builder(&self) -> Self::ConfigBuilder {
        Self::ConfigBuilder {
            stage_dims: Some(StageDims {
                lhs: StageDim {
                    tile_size_x: self.lhs_tile_size_x,
                    tile_size_y: self.lhs_tile_size_y,
                    num_tiles_x: self.lhs_num_tiles_x.unwrap_or(1),
                    num_tiles_y: self.lhs_num_tiles_y.unwrap_or(1),
                },
                rhs: StageDim {
                    tile_size_x: self.rhs_tile_size_x,
                    tile_size_y: self.rhs_tile_size_y,
                    num_tiles_x: self.rhs_num_tiles_x.unwrap_or(1),
                    num_tiles_y: self.rhs_num_tiles_y.unwrap_or(1),
                },
                out: StageDim {
                    tile_size_x: self.out_tile_size_x,
                    tile_size_y: self.out_tile_size_y,
                    num_tiles_x: self.out_num_tiles_x.unwrap_or(1),
                    num_tiles_y: self.out_num_tiles_y.unwrap_or(1),
                },
            }),
            ..Default::default()
        }
    }
}

impl MatmulConfig for CmmaConfig {
    type PreConfig = CmmaPreConfig;

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

impl CmmaConfig {
    pub fn stage_num_elems(&self, ident: Ident) -> u32 {
        self.stage_dim(ident).num_elements()
    }

    pub fn tile_num_elems(&self, ident: Ident) -> u32 {
        self.stage_dim(ident).tile_num_elements()
    }

    pub fn stage_dim(&self, ident: Ident) -> StageDim {
        match ident {
            Ident::Lhs => self.stage_dims.lhs,
            Ident::Rhs => self.stage_dims.rhs,
            Ident::Out => self.stage_dims.out,
        }
    }

    pub fn line_size(&self, ident: Ident) -> u32 {
        match ident {
            Ident::Lhs => self.line_sizes.0,
            Ident::Rhs => self.line_sizes.1,
            Ident::Out => self.line_sizes.2,
        }
    }

    pub fn layout(&self, ident: Ident) -> MatrixLayout {
        match ident {
            Ident::Lhs => self.layouts.0,
            Ident::Rhs => self.layouts.1,
            Ident::Out => MatrixLayout::RowMajor,
        }
    }
}
