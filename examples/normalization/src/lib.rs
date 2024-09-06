use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn norm_test<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = F::normalize(input[ABSOLUTE_POS]);
        output[ABSOLUTE_POS] = F::normalize(input[ABSOLUTE_POS]);
    }
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input = &[-1., 0., 1., 5.];
    let output_handle = client.empty(input.len() * core::mem::size_of::<f32>());
    let input_handle = client.create(f32::as_bytes(input));

    unsafe {
        norm_test::launch_unchecked::<F32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(input.len() as u32, 1, 1),
            ArrayArg::from_raw_parts(&input_handle, input.len(), 4),
            ArrayArg::from_raw_parts(&output_handle, input.len(), 4),
        )
    };

    let bytes = client.read(output_handle.binding());
    let output = f32::from_bytes(&bytes);

    println!(
        "Executed normalize with runtime {:?} => {output:?}",
        R::name()
    );
}
