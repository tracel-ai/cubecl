use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
struct MyType;

#[cube(debug)]
impl MyType {
    fn shape(self) -> u32 {
        0u32
    }
}
