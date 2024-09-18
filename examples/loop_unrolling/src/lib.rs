use cubecl::{prelude::*, OutputInfo};

#[cube(launch_unchecked)]
fn sum_basic<F: Float>(input: &Array<F>, output: &mut Array<F>, #[comptime] end: Option<u32>) {
    let unroll = end.is_some();
    let end = end.unwrap_or_else(|| input.len());

    let mut sum = F::new(0.0);

    #[unroll(unroll)]
    for i in 0..end {
        sum += input[i];
    }

    output[UNIT_POS] = sum;
}

#[cube(launch_unchecked)]
fn sum_subgroup<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    #[comptime] subgroup: bool,
    #[comptime] end: Option<u32>,
) {
    if subgroup {
        output[UNIT_POS] = subcube_sum(input[UNIT_POS]);
    } else {
        sum_basic(input, output, end);
    }
}

pub fn basic<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input = &[-1., 10., 1., 5.];

    let output_handle = client.empty(input.len() * core::mem::size_of::<f32>());
    let input_handle = client.create(f32::as_bytes(input));

    unsafe {
        sum_subgroup::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(input.len() as u32, 1, 1),
            ArrayArg::from_raw_parts(&input_handle, input.len(), 1),
            ArrayArg::from_raw_parts(&output_handle, input.len(), 1),
            client.features().enabled(cubecl::Feature::Subcube),
            Some(input.len() as u32),
        );
    }

    let bytes = client.read(output_handle.binding());
    let output = f32::from_bytes(&bytes);

    println!("Executed sum with runtime {:?} => {output:?}", R::name());
}
