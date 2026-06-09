use cubecl::{num_traits::One, prelude::*};

#[cube(launch_unchecked)]
/// A [Vector] represents a contiguous series of elements where SIMD operations may be available.
/// The runtime will automatically use SIMD instructions when possible for improved performance.
fn gelu_array<F: Float, N: Size>(input: &[Vector<F, N>], output: &mut [Vector<F, N>]) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = gelu_scalar(input[ABSOLUTE_POS]);
    }
}

#[cube]
fn gelu_scalar<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    // Execute the sqrt function at comptime.
    let sqrt2 = F::new(comptime!(2.0f32.sqrt()));
    let tmp = x / Vector::new(sqrt2);

    x * (Vector::erf(tmp) + Vector::one()) / Vector::new(F::new(2.0f32))
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input = &[-1., 0., 1., 5.];
    let vector_size = 4;
    let output_handle = client.empty(input.len() * core::mem::size_of::<f32>());
    let input_handle = client.create_from_slice(f32::as_bytes(input));

    unsafe {
        gelu_array::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(input.len() as u32 / vector_size as u32),
            vector_size,
            BufferArg::from_raw_parts(input_handle, input.len()),
            BufferArg::from_raw_parts(output_handle.clone(), input.len()),
        )
    };

    let bytes = client.read_one(output_handle).unwrap();
    let output = f32::from_bytes(&bytes);

    // Should be [-0.1587,  0.0000,  0.8413,  5.0000]
    println!(
        "Executed gelu with runtime {:?} => {output:?}",
        R::name(&client)
    );
}
