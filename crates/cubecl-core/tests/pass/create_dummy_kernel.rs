//! `create_dummy_kernel` generates a whole function body — argument
//! registration included — that nothing else in the workspace instantiates.
//! Compiling it here is what keeps that body honest.
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube(launch, create_dummy_kernel)]
fn generic_kernel<F: Float>(input: &[F], output: &mut [F], #[comptime] factor: u32) {
    if UNIT_POS == 0 {
        output[0] = input[0] * F::cast_from(factor);
    }
}

#[cube(launch, create_dummy_kernel)]
fn plain_kernel(output: &mut [f32]) {
    if UNIT_POS == 0 {
        output[0] = 5.0;
    }
}

fn main() {
    // Naming the items checks their signatures, which the generated bodies
    // alone would not.
    let _ = generic_kernel::create_dummy_kernel::<f32>;
    let _ = plain_kernel::create_dummy_kernel;
}
