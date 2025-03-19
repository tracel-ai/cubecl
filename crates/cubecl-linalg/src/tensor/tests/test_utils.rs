use cubecl_core::{CubeElement, prelude::Numeric};

pub(crate) fn identity_cpu<E: Numeric + CubeElement>(dim: usize) -> Vec<E> {
    let num_elements = dim * dim;
    let mut result = vec![E::from_int(0); num_elements];

    let mut pos = 0usize;
    while pos < num_elements {
        result[pos] = E::from_int(1);
        pos += dim + 1;
    }

    result
}
