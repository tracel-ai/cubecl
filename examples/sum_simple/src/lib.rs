use cubecl::prelude::*;

const VECTORIZATION: u8 = 16;

#[cube(launch_unchecked)]
/// A [Line] represents a contiguous series of elements where SIMD operations may be available.
/// The runtime will automatically use SIMD instructions when possible for improved performance.
fn sum_simple<F: Float>(
    input_a: &Array<Line<F>>,
    input_b: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
) {
    output[ABSOLUTE_POS] = input_a[ABSOLUTE_POS] + input_b[ABSOLUTE_POS];
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input_a: [f32; 128] = std::array::from_fn(|i| i as f32 * 0.5);
    let input_b: [f32; 128] = std::array::from_fn(|i| i as f32 * 0.5);
    let output_handle = client.empty(input_a.len() * core::mem::size_of::<f32>());
    let input_a_handle = client.create(f32::as_bytes(&input_a));
    let input_b_handle = client.create(f32::as_bytes(&input_b));

    unsafe {
        sum_simple::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 2, 2),
            CubeDim::new(1, 4, 1),
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
