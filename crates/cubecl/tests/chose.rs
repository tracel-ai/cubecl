use cubecl::{CubeDim, CubeElement, Runtime, TestRuntime, cube, prelude::*};

#[cube(launch)]
fn test_chose_cube(lhs: f32, rhs: f32, out: &mut [f32]) {
    if UNIT_POS == 0 {
        out[0] = lhs + rhs + lhs;
        out[0] = lhs + rhs;
        out[0] = lhs + rhs;
    }
}

#[test]
fn test_chose_1() {
    let client = TestRuntime::client(&Default::default());

    let lhs = 1.0;
    let rhs = 2.0;
    let output = client.empty(core::mem::size_of::<f32>());

    test_chose_cube::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        lhs,
        rhs,
        unsafe { BufferArg::from_raw_parts(output.clone(), 1) },
    );

    let bytes = client.read_one(output).unwrap();
    let result = f32::from_bytes(&bytes);

    println!("{:#?}", client.metrics().read().unwrap());

    assert_eq!(result[0], lhs + rhs);
}

#[cube(launch)]
fn test_chose_cube_tensor(lhs: &Tensor<f32>, rhs: &Tensor<f32>, out: &mut Tensor<f32>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] + rhs[ABSOLUTE_POS];
    }
}

#[test]
fn test_chose_tensor() {
    let client = TestRuntime::client(&Default::default());

    let lhs = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0]));
    let rhs = client.create_from_slice(f32::as_bytes(&[10.0, 20.0, 30.0]));
    let output = client.empty(3 * core::mem::size_of::<f32>());

    test_chose_cube_tensor::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(3),
        // from_raw_parts(handle, strides, shape) — contiguous 1D tensor of length 3
        unsafe { TensorArg::from_raw_parts(lhs, [1].into(), [3].into()) },
        unsafe { TensorArg::from_raw_parts(rhs, [1].into(), [3].into()) },
        unsafe { TensorArg::from_raw_parts(output.clone(), [1].into(), [3].into()) },
    );

    let bytes = client.read_one(output).unwrap();
    let result = f32::from_bytes(&bytes);

    assert_eq!(result, &[11.0, 22.0, 33.0]);
}

#[cube(launch)]
fn test_chose_cube_shape(input: &Tensor<f32>, out: &mut Tensor<u32>) {
    if UNIT_POS == 0 {
        out[0] = input.shape(0) as u32;
        out[1] = input.shape(1) as u32;
    }
}

#[test]
fn test_chose_shape() {
    let client = TestRuntime::client(&Default::default());
    let input = client.create_from_slice(f32::as_bytes(&[0.0; 12]));
    let output = client.empty(2 * core::mem::size_of::<u32>());

    test_chose_cube_shape::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { TensorArg::from_raw_parts(input, [4, 1].into(), [3, 4].into()) },
        unsafe { TensorArg::from_raw_parts(output.clone(), [2, 1].into(), [1, 2].into()) },
    );

    let bytes = client.read_one(output).unwrap();
    let result = u32::from_bytes(&bytes);
    // input shape is [3, 4]
    assert_eq!(result, &[3, 4]);
}

#[cube(launch)]
fn test_chose_vectorized<F: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    scalar: f32,
    out: &mut Tensor<Vector<F, N>>,
) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS] + Vector::cast_from(scalar);
    }
}

#[test]
fn test_chose_cube_vectorized() {
    let client = TestRuntime::client(&Default::default());

    let vectorization = 2;
    let input = client.create_from_slice(f32::as_bytes(&[
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
    ]));
    let output = client.empty(12 * core::mem::size_of::<f32>());

    test_chose_vectorized::launch::<f32, TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(6),
        vectorization,
        unsafe { TensorArg::from_raw_parts(input, [4, 1].into(), [3, 4].into()) },
        10.0,
        unsafe { TensorArg::from_raw_parts(output.clone(), [4, 1].into(), [3, 4].into()) },
    );

    let bytes = client.read_one(output).unwrap();
    let result = f32::from_bytes(&bytes);

    println!("{:#?}", client.metrics().read().unwrap());

    assert_eq!(
        result,
        &[11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22.]
    );
}
