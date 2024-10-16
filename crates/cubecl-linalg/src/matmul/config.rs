use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::fmt::Debug;
use std::hash::Hash;

#[cube]
pub trait PlaneMapper {
    fn plane_id() -> u32;
    fn plane_unit() -> u32;
    fn num_planes() -> u32;
    fn plane_dim() -> u32;
}

#[cube]
pub trait SubRoutine {
    type ProblemDefinition: CubeType;

    fn assert_can_process(problem: Self::ProblemDefinition);
}

pub trait Config:
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
    type ProblemDefinition;

    fn from_problem(problem: Self::ProblemDefinition) -> Self;
}

pub struct Requirements {
    pub min_planes: u32,
    pub max_planes: u32,
    pub num_cubes: u32,
}
