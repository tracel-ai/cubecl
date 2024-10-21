use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::fmt::Debug;
use std::hash::Hash;

#[cube]
pub trait PlaneMapper {
    fn plane_id() -> u32;
    fn plane_unit() -> u32;
}

pub trait MatmulConfig: ComptimeConfig {}

pub trait ComptimeConfig:
    CubeType + Copy + Clone + Send + Sync + 'static + Eq + PartialEq + Hash + Debug + IntoRuntime
{
}
