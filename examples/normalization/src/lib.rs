use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn norm_test<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    if ABSOLUTE_POS < input.len() {
        let original_magnitude = F::magnitude(input[ABSOLUTE_POS]);
        let normalized = F::normalize(input[ABSOLUTE_POS]);
        let new_magnitude = F::magnitude(normalized);
        output[ABSOLUTE_POS] = original_magnitude + new_magnitude;
    }
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input = &[-1., 0., 1., 5.];
    let output_handle = client.empty(2 * core::mem::size_of::<f32>());
    let input_handle = client.create(f32::as_bytes(input));

    unsafe {
        norm_test::launch_unchecked::<F32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(input.len() as u32, 1, 1),
            ArrayArg::from_raw_parts(&input_handle, input.len(), 2),
            ArrayArg::from_raw_parts(&output_handle, input.len(), 1),
        )
    };

    let bytes = client.read(output_handle.binding());
    let output = f32::from_bytes(&bytes);

    println!(
        "Executed normalize with runtime {:?} => {output:?}",
        R::name()
    );
}
