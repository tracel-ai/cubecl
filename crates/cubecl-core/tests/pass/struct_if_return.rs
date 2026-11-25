use cubecl::prelude::*;
use cubecl_core as cubecl;

// Regression for #333: both arms of a runtime if must be able to build a
// CubeType and return it.
#[derive(CubeType)]
struct Foo {
    a: f32,
}

#[cube]
impl Foo {
    pub fn foo(&self, v: u32) -> Foo {
        if v > 0 {
            Foo { a: 1.0 }
        } else {
            Foo { a: 0.0 }
        }
    }
}

fn main() {}
