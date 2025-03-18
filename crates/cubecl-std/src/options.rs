use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeLaunch, CubeType)]
pub enum CubeOption<T: CubeLaunch> {
    Some(T),
    None,
}

#[derive(CubeLaunch, CubeType)]
pub struct A<T: CubeLaunch> {
    pub x: T,
}
