use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::fmt::Debug;
use std::hash::Hash;

#[cube]
pub trait PlaneMapper {
    fn plane_id() -> u32;
    fn plane_unit() -> u32;
}

pub trait MatmulConfig: ComptimeConfig {
    // fn num_planes(&self) -> u32;
    // fn plane_dim(&self) -> u32;
    // fn cube_dim(&self) -> CubeDim;
    // fn cube_count(&self) -> CubeCount;
}

pub trait MatmulLaunchConfig {
    fn cube_dim(&self) -> CubeDim;
    // {
    //     CubeDim::new(self.cube_count.0, self.cube_count.1, self.cube_count.2)
    // }

    fn cube_count(&self) -> CubeCount;
    //  {
    //     CubeCount::Static(self.cube_count.0, self.cube_count.1, self.cube_count.2)
    // }
}

pub trait ComptimeConfig:
    CubeType + Copy + Clone + Send + Sync + 'static + Eq + PartialEq + Hash + Debug + IntoRuntime
{
}
