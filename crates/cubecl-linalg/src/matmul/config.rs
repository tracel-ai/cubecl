use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::fmt::Debug;
use std::hash::Hash;

use super::problem::MatmulProblem;

#[cube]
pub trait PlaneMapper {
    fn plane_id() -> u32;
    fn plane_unit() -> u32;
}

pub trait MatmulConfig:
    CubeType<ExpandType = Self>
    + Copy
    + Clone
    + Send
    + Sync
    + 'static
    + Init
    + Eq
    + PartialEq
    + Hash
    + Debug
{
    type ConfigBuilder: ConfigBuilder<Config = Self>;

    fn build() -> Self::ConfigBuilder;
    fn num_planes(&self) -> u32;
    fn plane_dim(&self) -> u32;
}

pub trait ConfigBuilder {
    type Config: MatmulConfig;

    fn from_cube_settings(&self, cube_dim: &CubeDim, cube_count: &CubeCount) -> Self;
    fn from_problem(&self, problem: &MatmulProblem) -> Self::Config;
}
