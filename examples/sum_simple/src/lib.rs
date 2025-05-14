use cubecl::prelude::*;

const VECTORIZATION: u8 = 4;

#[cube(launch_unchecked)]
/// A [Line] represents a contiguous series of elements where SIMD operations may be available.
/// The runtime will automatically use SIMD instructions when possible for improved performance.
fn gelu_array<F: Float>(input_a: &Array<F>, input_b: &Array<F>, output: &mut Array<F>) {
    if ABSOLUTE_POS < input_a.len() {
        output[ABSOLUTE_POS] = input_a[ABSOLUTE_POS] + input_b[ABSOLUTE_POS];
    }
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input_a: [f32; 128] = std::array::from_fn(|i| i as f32);
    let input_b: [f32; 128] = std::array::from_fn(|i| (i * i) as f32);
    let output_handle = client.empty(input_a.len() * core::mem::size_of::<f32>());
    let input_a_handle = client.create(f32::as_bytes(&input_a));
    let input_b_handle = client.create(f32::as_bytes(&input_b));

    unsafe {
        gelu_array::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new((input_a.len() / VECTORIZATION as usize) as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input_a_handle, input_a.len(), VECTORIZATION),
            ArrayArg::from_raw_parts::<f32>(&input_b_handle, input_a.len(), VECTORIZATION),
            ArrayArg::from_raw_parts::<f32>(&output_handle, input_a.len(), VECTORIZATION),
        )
    };

    let bytes = client.read_one(output_handle.binding());
    let output = f32::from_bytes(&bytes);

    println!(
        "Executed sum_simple with runtime {:?} => {output:?}",
        R::name(&client)
    );
}
