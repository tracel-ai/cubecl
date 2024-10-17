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

pub trait MatmulConfig: ComptimeConfig {
    fn num_planes(&self) -> u32;
    fn plane_dim(&self) -> u32;

    // Create a valid configuration with the default settings given basic launching informations
    fn default(cube_dim: CubeDim, cube_count: CubeCount, problem: MatmulProblem);
}

pub trait ComptimeConfig:
    CubeType + Copy + Clone + Send + Sync + 'static + Eq + PartialEq + Hash + Debug + IntoRuntime
{
}
