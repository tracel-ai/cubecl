use cubecl::prelude::*;

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
    type Primitive = half::f16;
    type CubeType = F16;

    let client = R::client(device);
    let input = &[-1., 0., 1., 5.].map(|f| Primitive::from_f32(f));

    let output_handle = client.empty(input.len() * core::mem::size_of::<Primitive>());
    let input_handle = client.create(Primitive::as_bytes(input));

    gelu_array::launch::<CubeType, R>(
        client.clone(),
        CubeCount::Static(1, 1, 1),
        CubeDim::new(input.len() as u32 / 4, 1, 1),
        ArrayArg::vectorized(4, &input_handle, input.len()),
        ArrayArg::vectorized(4, &output_handle, input.len()),
    );

    let bytes = client.read(output_handle.binding());
    let output = Primitive::from_bytes(&bytes);

    // Should be [-0.1587,  0.0000,  0.8413,  5.0000]
    println!("Executed gelu with runtime {:?} => {output:?}", R::name());
}
