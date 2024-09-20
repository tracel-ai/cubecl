use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn gelu_array<F: Float>(inputs: &Sequence<Array<F>>, output: &mut Array<F>) {
    #[unroll]
    for i in 0..inputs.len() {
        let array = inputs.index(i);
        if ABSOLUTE_POS < array.len() {
            output[ABSOLUTE_POS] = gelu_scalar::<F>(array[ABSOLUTE_POS]);
        }
    }
}

#[cube]
fn gelu_scalar<F: Float>(x: F) -> F {
    x * (F::erf(x / F::new(2.0f32.sqrt())) + F::new(1.0)) / F::new(2.0)
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input = &[-1., 0., 1., 5.];
    let vectorization = 4;
    let output_handle = client.empty(input.len() * core::mem::size_of::<f32>());
    let input_handle = client.create(f32::as_bytes(input));

    let mut sequence = SequenceArg::new();
    unsafe {
        sequence.push(ArrayArg::from_raw_parts(
            &input_handle,
            input.len(),
            vectorization as u8,
        ));

        gelu_array::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(input.len() as u32 / vectorization, 1, 1),
            sequence,
            ArrayArg::from_raw_parts(&output_handle, input.len(), vectorization as u8),
        )
    };

    let bytes = client.read(output_handle.binding());
    let output = f32::from_bytes(&bytes);

    // Should be [-0.1587,  0.0000,  0.8413,  5.0000]
    println!("Executed gelu with runtime {:?} => {output:?}", R::name());
}
