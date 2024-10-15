use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::hash::Hash;
use std::fmt::Debug;

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
}
