use cubecl_core::{prelude::Numeric, CubeElement};

pub(crate) fn eye_cpu<E: Numeric + CubeElement>(dim: u32) -> Vec<E> {
    let dim_usize = usize::try_from(dim).expect("expected dimension too large");
    let num_elements = dim_usize * dim_usize;
    let mut result = vec![E::from_int(0); num_elements];

    let mut pos = 0usize;
    while pos < num_elements {
        result[pos] = E::from_int(1);
        pos += dim_usize + 1;
    }

    result
}
