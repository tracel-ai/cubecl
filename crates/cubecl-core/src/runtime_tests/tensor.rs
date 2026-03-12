use alloc::vec;

use crate as cubecl;
use cubecl::prelude::*;

#[cube(launch)]
pub fn tensor_coordinate<N: Size>(input: &Tensor<Vector<f32, N>>, output: &mut Array<u32>) {
    let index = UNIT_POS_X as usize;
    let dim = UNIT_POS_Y as usize;
    output[UNIT_POS as usize] = input.coordinate(index, dim) as u32;
}

pub fn test_tensor_coordinate<R: Runtime>(client: ComputeClient<R>) {
    let stride = [2, 1, 4];
    let shape = [2, 2, 3];

    let input_size = shape.iter().product::<usize>();
    let input = client.empty(core::mem::size_of::<f32>() * input_size);

    // Each column corresponds to a complete coordinate.
    // That is, when increasing the index, the coordinates are
    // [0,0,0], [0,1,0] ... [1,1,2].
    let expected = vec![
        0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, //
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, //
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, //
    ];

    let output_size = shape.len() * input_size;

    // The result is independent of the vector size
    for vector_size in client.io_optimized_vector_sizes(size_of::<f32>()) {
        let output = client.empty(core::mem::size_of::<u32>() * output_size);
        unsafe {
            tensor_coordinate::launch(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(input_size as u32, shape.len() as u32),
                vector_size,
                TensorArg::from_raw_parts(input.clone(), stride.into(), shape.into()),
                ArrayArg::from_raw_parts(output.clone(), output_size),
            )
        };

        let actual = client.read_one_unchecked(output);
        let actual = u32::from_bytes(&actual);

        assert_eq!(actual, expected);
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_tensor_indexing {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_tensor_coordinate() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::tensor::test_tensor_coordinate::<TestRuntime>(client);
        }
    };
}
