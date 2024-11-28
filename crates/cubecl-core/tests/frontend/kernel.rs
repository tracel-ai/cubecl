use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube(launch)]
pub fn with_kernel<F: Float>(kernel: &mut Array<F>) {
    if ABSOLUTE_POS > kernel.len() {
        kernel[ABSOLUTE_POS] = F::cast_from(5.0);
    }
}
