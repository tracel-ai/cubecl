use cubecl::prelude::*;

// These functions aren't implemented on Vector, need to fix this at some point
#[cube(launch_unchecked)]
fn norm_test<F: Float, N: Size>(
    input: &[Vector<F, N>],
    output_a: &mut [Vector<F, N>],
    output_b: &mut [Vector<F, N>],
) {
    if ABSOLUTE_POS < input.len() {
        output_a[ABSOLUTE_POS] = Vector::cast_from(F::normalize(F::cast_from(input[ABSOLUTE_POS])));
        output_b[ABSOLUTE_POS] = input[ABSOLUTE_POS]
            / Vector::cast_from(F::magnitude(F::cast_from(input[ABSOLUTE_POS])));
    }
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input = &[-1., 0., 1., 5.];
    let input_handle = client.create_from_slice(f32::as_bytes(input));
    let output_a_handle = client.empty(input.len() * core::mem::size_of::<f32>());
    let output_b_handle = client.empty(input.len() * core::mem::size_of::<f32>());

    unsafe {
        norm_test::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(input.len() as u32),
            4,
            BufferArg::from_raw_parts(input_handle, input.len()),
            BufferArg::from_raw_parts(output_a_handle.clone(), input.len()),
            BufferArg::from_raw_parts(output_b_handle.clone(), input.len()),
        )
    };

    let bytes = client.read_one(output_a_handle).unwrap();
    let output = f32::from_bytes(&bytes);

    println!(
        "Executed normalize with runtime {:?} => {output:?}",
        R::name(&client)
    );

    let bytes = client.read_one(output_b_handle).unwrap();
    let output = f32::from_bytes(&bytes);

    println!(
        "Executed normalize using magnitude with runtime {:?} => {output:?}",
        R::name(&client)
    );
}
