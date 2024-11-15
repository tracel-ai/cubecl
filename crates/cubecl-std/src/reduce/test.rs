#![allow(missing_docs)]

use cubecl_core::{prelude::*, Feature};

use crate::reduce::sum::{reduce_sum, ReduceConfig};

#[macro_export]
macro_rules! testgen_reduce {
    () => {
        use super::*;
        use cubecl_core::{CubeCount, CubeDim};
        use cubecl_std::reduce::test::{impl_reduce_sum_test, TestCase, TestTensorParts};

        #[test]
        pub fn reduce_sum_vector_single_plane() {
            impl_reduce_sum_test::<TestRuntime, u32>(
                &Default::default(),
                TestCase {
                    input: TestTensorParts {
                        values: (0..32).collect(),
                        stride: vec![1],
                        shape: vec![32],
                    },
                    output: TestTensorParts {
                        values: vec![0],
                        stride: vec![1],
                        shape: vec![1],
                    },
                    line_size: 1,
                    expected: vec![496],
                    tolerance: None,
                    cube_count: CubeCount::Static(1, 1, 1),
                    cube_dim: CubeDim::new(32, 1, 1),
                    sum_dim: 0,
                },
            )
        }

        #[test]
        pub fn reduce_sum_vector_single_plane_line_size_four() {
            impl_reduce_sum_test::<TestRuntime, u32>(
                &Default::default(),
                TestCase {
                    input: TestTensorParts {
                        values: (0..32).collect(),
                        stride: vec![1],
                        shape: vec![32],
                    },
                    output: TestTensorParts {
                        values: vec![0, 0, 0, 0],
                        stride: vec![1],
                        shape: vec![1],
                    },
                    line_size: 4,
                    expected: vec![112, 120, 128, 136],
                    tolerance: None,
                    cube_count: CubeCount::Static(1, 1, 1),
                    cube_dim: CubeDim::new(32, 1, 1),
                    sum_dim: 0,
                },
            )
        }

        #[test]
        pub fn reduce_sum_vector_long_single_plane() {
            impl_reduce_sum_test::<TestRuntime, u32>(
                &Default::default(),
                TestCase {
                    input: TestTensorParts {
                        values: (0..128).collect(),
                        stride: vec![1],
                        shape: vec![128],
                    },
                    output: TestTensorParts {
                        values: vec![0],
                        stride: vec![1],
                        shape: vec![1],
                    },
                    line_size: 1,
                    expected: vec![8128],
                    tolerance: None,
                    cube_count: CubeCount::Static(1, 1, 1),
                    cube_dim: CubeDim::new(32, 1, 1),
                    sum_dim: 0,
                },
            )
        }

        #[test]
        pub fn reduce_sum_long_vector_four_planes() {
            impl_reduce_sum_test::<TestRuntime, u32>(
                &Default::default(),
                TestCase {
                    input: TestTensorParts {
                        values: (0..128).collect(),
                        stride: vec![1],
                        shape: vec![128],
                    },
                    output: TestTensorParts {
                        values: vec![0],
                        stride: vec![1],
                        shape: vec![1],
                    },
                    line_size: 1,
                    expected: vec![8128],
                    tolerance: None,
                    cube_count: CubeCount::Static(1, 1, 1),
                    cube_dim: CubeDim::new(32, 4, 1),
                    sum_dim: 0,
                },
            )
        }

        #[test]
        pub fn reduce_sum_vector_with_remainder_single_plane() {
            impl_reduce_sum_test::<TestRuntime, u32>(
                &Default::default(),
                TestCase {
                    input: TestTensorParts {
                        values: (0..100).collect(),
                        stride: vec![1],
                        shape: vec![100],
                    },
                    output: TestTensorParts {
                        values: vec![0],
                        stride: vec![1],
                        shape: vec![1],
                    },
                    line_size: 1,
                    expected: vec![4950],
                    tolerance: None,
                    cube_count: CubeCount::Static(1, 1, 1),
                    cube_dim: CubeDim::new(32, 1, 1),
                    sum_dim: 0,
                },
            )
        }

        #[test]
        pub fn reduce_sum_vector_with_remainder_four_planes() {
            impl_reduce_sum_test::<TestRuntime, u32>(
                &Default::default(),
                TestCase {
                    input: TestTensorParts {
                        values: (0..100).collect(),
                        stride: vec![1],
                        shape: vec![100],
                    },
                    output: TestTensorParts {
                        values: vec![0],
                        stride: vec![1],
                        shape: vec![1],
                    },
                    line_size: 1,
                    expected: vec![4950],
                    tolerance: None,
                    cube_count: CubeCount::Static(1, 1, 1),
                    cube_dim: CubeDim::new(32, 4, 1),
                    sum_dim: 0,
                },
            )
        }

        #[test]
        pub fn reduce_sum_vector_f32_eight_planes() {
            impl_reduce_sum_test::<TestRuntime, f32>(
                &Default::default(),
                TestCase {
                    input: TestTensorParts {
                        values: (0..1024).map(|n| n as f32).collect(),
                        stride: vec![1],
                        shape: vec![1024],
                    },
                    output: TestTensorParts {
                        values: vec![0.0],
                        stride: vec![1],
                        shape: vec![1],
                    },
                    line_size: 1,
                    expected: vec![523776.0],
                    tolerance: Some(1e-9),
                    cube_count: CubeCount::Static(1, 1, 1),
                    cube_dim: CubeDim::new(32, 8, 1),
                    sum_dim: 0,
                },
            )
        }

        #[test]
        pub fn reduce_sum_vector_f32_too_many_planes() {
            impl_reduce_sum_test::<TestRuntime, f32>(
                &Default::default(),
                TestCase {
                    input: TestTensorParts {
                        values: (0..128).map(|n| n as f32).collect(),
                        stride: vec![1],
                        shape: vec![128],
                    },
                    output: TestTensorParts {
                        values: vec![0.0],
                        stride: vec![1],
                        shape: vec![1],
                    },
                    line_size: 1,
                    expected: vec![8128.0],
                    tolerance: Some(1e-9),
                    cube_count: CubeCount::Static(1, 1, 1),
                    cube_dim: CubeDim::new(32, 8, 1),
                    sum_dim: 0,
                },
            )
        }
    };
}

#[derive(Debug)]
pub struct TestCase<N> {
    pub input: TestTensorParts<N>,
    pub output: TestTensorParts<N>,
    pub line_size: u8,
    pub expected: Vec<N>,
    pub tolerance: Option<f32>,
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
    pub sum_dim: u32,
}

#[derive(Debug)]
pub struct TestTensorParts<N> {
    pub values: Vec<N>,
    pub stride: Vec<usize>,
    pub shape: Vec<usize>,
}

pub fn impl_reduce_sum_test<R: Runtime, N: Numeric + CubeElement + std::fmt::Display>(
    device: &R::Device,
    test: TestCase<N>,
) {
    let client = R::client(device);
    if !client.properties().feature_enabled(Feature::Plane) {
        // Can't execute the test.
        return;
    }

    let input_handle = client.create(N::as_bytes(&test.input.values));
    let output_handle = client.create(N::as_bytes(&test.output.values));

    let config = ReduceConfig {
        line_size: test.line_size as u32,
        plane_size: test.cube_dim.x,
        num_planes: test.cube_dim.y,
    };

    unsafe {
        let input_tensor = TensorArg::from_raw_parts::<N>(
            &input_handle,
            &test.input.stride,
            &test.input.shape,
            test.line_size,
        );
        let output_tensor = TensorArg::from_raw_parts::<N>(
            &output_handle,
            &test.output.stride,
            &test.output.shape,
            test.line_size,
        );

        reduce_sum::launch_unchecked::<N, R>(
            &client,
            test.cube_count,
            test.cube_dim,
            input_tensor,
            output_tensor,
            config,
        );
    }

    let binding = output_handle.binding();
    let bytes = client.read(binding);
    let output_values = N::from_bytes(&bytes);

    match test.tolerance {
        Some(tolerance) => assert_approx_equal_abs(output_values, &test.expected, tolerance),
        None => assert_eq!(output_values, test.expected),
    }
}

pub fn assert_approx_equal_abs<N: Numeric>(actual: &[N], expected: &[N], epsilon: f32) {
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a = a.to_f32().unwrap();
        let e = e.to_f32().unwrap();
        let diff = (a - e).abs();
        assert!(diff < epsilon, "Values differ more than epsilon: index={} actual={}, expected={}, difference={}, epsilon={}",
            i,
            a,
            e,
            diff,
            epsilon
            );
    }
}
