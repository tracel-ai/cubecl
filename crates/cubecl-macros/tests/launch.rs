use cubecl_core::new_ir::{element::Tensor1, ABSOLUTE_POS};
use cubecl_core::prelude::*;

mod common;

#[test]
fn launch_unchecked_simple() {
    #[allow(unused)]
    #[cube(launch_unchecked)]
    fn copy_tensor(input: &Tensor1<f32>, output: &mut Tensor1<f32>) {
        let idx = ABSOLUTE_POS;
        output[idx] = input[idx];
    }
}

#[test]
fn launch_unchecked_simple_2() {}
