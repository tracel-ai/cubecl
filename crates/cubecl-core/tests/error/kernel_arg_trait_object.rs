use cubecl::prelude::*;
use cubecl_core as cubecl;

trait Foo {}

#[cube]
fn kernel_arg_trait_object(x: &dyn Foo) {
    let _ = x;
}

fn main() {}
