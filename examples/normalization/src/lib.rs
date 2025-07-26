use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn norm_test<F: Float>(input: &Array<F>, output_a: &mut Array<F>, output_b: &mut Array<F>) {
    if ABSOLUTE_POS < input.len() {
        output_a[ABSOLUTE_POS] = F::normalize(input[ABSOLUTE_POS]);
        output_b[ABSOLUTE_POS] = input[ABSOLUTE_POS] / F::magnitude(input[ABSOLUTE_POS]);
    }
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input = &[-1., 0., 1., 5.];
    let input_handle = client
        .create(f32::as_bytes(input))
        .expect("Failed to allocate memory");
    let output_a_handle = client
        .empty(input.len() * core::mem::size_of::<f32>())
        .expect("Failed to allocate memory");
    let output_b_handle = client
        .empty(input.len() * core::mem::size_of::<f32>())
        .expect("Failed to allocate memory");

    unsafe {
        norm_test::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(input.len() as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input_handle, input.len(), 4),
            ArrayArg::from_raw_parts::<f32>(&output_a_handle, input.len(), 4),
            ArrayArg::from_raw_parts::<f32>(&output_b_handle, input.len(), 4),
        )
    };

    let bytes = client.read_one(output_a_handle.binding());
    let output = f32::from_bytes(&bytes);

    println!(
        "Executed normalize with runtime {:?} => {output:?}",
        R::name(&client)
    );

    let bytes = client.read_one(output_b_handle.binding());
    let output = f32::from_bytes(&bytes);

    println!(
        "Executed normalize using magnitude with runtime {:?} => {output:?}",
        R::name(&client)
    );
}
