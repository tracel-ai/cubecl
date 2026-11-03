use cubecl::{
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
};
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::support::{
    BaseInputSpec,
    test_tensor::{custom::cast_f32_to_dtype, strides::physical_extent},
};

/// Uniform values in `[lo, hi]` over the whole physical buffer the spec's strides reach, which
/// is more than the shape's product for jumpy strides and less for broadcast ones.
pub(crate) fn build_uniform(spec: BaseInputSpec, seed: u64, lo: f32, hi: f32) -> TensorHandle {
    let physical_len = physical_extent(&spec.shape, &spec.strides());

    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..physical_len)
        .map(|_| lo + (hi - lo) * rng.random::<f32>())
        .collect();

    let handle = spec
        .client
        .create_from_slice(&cast_f32_to_dtype(&data, spec.dtype));
    TensorHandle::new(
        handle,
        Shape::from(vec![physical_len]),
        Strides::new(&[1]),
        spec.dtype,
    )
}
