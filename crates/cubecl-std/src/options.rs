use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeType)]
pub enum CubeOption<T: CubeType + Clone + IntoRuntime> {
    Some(T),
    None,
}
