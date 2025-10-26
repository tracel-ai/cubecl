use std::fmt::Display;

use cubecl_core::{CubeElement, Runtime, client::ComputeClient, prelude::Float, server};
use cubecl_std::tensor::{TensorHandle, into_contiguous};

pub(crate) fn tensorhandler_from_data<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R::Server>,
    shape: Vec<usize>,
    data: &[F],
) -> TensorHandle<R, F> {
    let handle = client.create(F::as_bytes(data));
    TensorHandle::new_contiguous(shape, handle)
}

pub(crate) fn transpose_matrix<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R::Server>,
    matrix: &mut TensorHandle<R, F>,
) -> TensorHandle<R, F> {
    matrix.strides.swap(1, 0);
    matrix.shape.swap(1, 0);

    into_contiguous::<R, F>(client, &matrix.as_ref())
}

/// Compares the content of a handle to a given slice of f32.
pub(crate) fn assert_equals_approx<R: Runtime, F: Float + CubeElement + Display>(
    client: &ComputeClient<R::Server>,
    output: server::Handle,
    shape: &[usize],
    strides: &[usize],
    expected: &[F],
    epsilon: f32,
) -> Result<(), String> {
    let actual = client.read_one_tensor(output.copy_descriptor(shape, strides, size_of::<F>()));
    let actual = F::from_bytes(&actual);

    // normalize to type epsilon
    let epsilon = (epsilon / f32::EPSILON * F::EPSILON.to_f32().unwrap()).max(epsilon);

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        // account for lower precision at higher values
        let allowed_error = (epsilon * e.to_f32().unwrap().abs()).max(epsilon);

        if f32::is_nan(a.to_f32().unwrap())
            || f32::abs(a.to_f32().unwrap() - e.to_f32().unwrap()) >= allowed_error
        {
            return Err(format!(
                "Values differ more than epsilon: index={} actual={}, expected={}, difference={}, epsilon={}",
                i,
                *a,
                *e,
                f32::abs(a.to_f32().unwrap() - e.to_f32().unwrap()),
                epsilon
            ));
        }
    }

    Ok(())
}
