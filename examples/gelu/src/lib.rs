use cubecl::{linalg::tensor::TensorHandle, prelude::*};

#[cube(launch)]
fn gelu_array<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = gelu_scalar::<F>(input[ABSOLUTE_POS]);
    }
}

#[cube]
fn gelu_scalar<F: Float>(x: F) -> F {
    x * (F::erf(x / F::sqrt(2.0.into())) + 1.0) / 2.0
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input = &[-1., 0., 1., 5.];
    let output = TensorHandle::<R, F32>::zeros(&client, vec![4]);
    let input_handle = client.create(f32::as_bytes(input));

    gelu_array::launch::<F32, R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(input.len() as u32, 1, 1),
        ArrayArg::new(&input_handle, input.len()),
        ArrayArg::new(&output.handle, input.len()),
    );

    let bytes = client.read(output.handle.binding());
    let output = f32::from_bytes(&bytes);

    // Should be [-0.1587,  0.0000,  0.8413,  5.0000]
    println!("Executed gelu with runtime {:?} => {output:?}", R::name());
}
