use crate as cubecl;
use cubecl::prelude::*;

#[cube(launch)]
pub fn tensor_coordinate(input: &Tensor<f32>, output: &mut Array<u32>) {
    let index = UNIT_POS_X;
    let dim = UNIT_POS_Y;
    output[UNIT_POS] = input.coordinate(index, dim);
}

pub fn test_tensor_coordinate<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let stride = [2, 1, 4];
    let shape = [2, 2, 3];

    let input_size = shape.iter().product::<usize>();
    let input = client
        .empty(core::mem::size_of::<f32>() * input_size)
        .expect("Alloc failed");

    // Each column corresponds to a complete coordinate.
    // That is, when increasing the index, the coordinates are
    // [0,0,0], [0,1,0] ... [1,1,2].
    let expected = vec![
        0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, //
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, //
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, //
    ];

    let output_size = shape.len() * input_size;

    // The result is independent of the line size
    for &line_size in R::supported_line_sizes() {
        let output = client
            .empty(core::mem::size_of::<u32>() * output_size)
            .expect("Alloc failed");
        unsafe {
            tensor_coordinate::launch::<R>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(input_size as u32, shape.len() as u32, 1),
                TensorArg::from_raw_parts::<f32>(&input, &stride, &shape, line_size),
                ArrayArg::from_raw_parts::<u32>(&output, output_size, 1),
            )
        };

        let actual = client.read_one(output.binding());
        let actual = u32::from_bytes(&actual);

        assert_eq!(actual, expected);
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_tensor_indexing {
    () => {
        use super::*;

        #[test]
        fn test_tensor_coordinate() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::tensor::test_tensor_coordinate::<TestRuntime>(client);
        }
    };
}
