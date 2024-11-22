#![allow(missing_docs)]

use cubecl_core::{prelude::*, Feature};

use crate::sum::{reduce_sum, reduce_sum_lined, ReduceConfig};

#[macro_export]
macro_rules! testgen_reduce {
    () => {
        use super::*;
        use cubecl_core::CubeCount;
        use cubecl_reduce::test::{impl_reduce_sum_test, TestCase, TestTensorParts};

        #[test]
        pub fn reduce_sum_vector_single_plane() {
            let test = TestCase::new(
                // input
                TestTensorParts::new_vector((0..32).collect()),
                // output
                TestTensorParts::new_vector(vec![0]),
                // expected
                vec![496],
            );
            impl_reduce_sum_test::<TestRuntime, u32>(&Default::default(), test);
        }

        #[test]
        pub fn reduce_sum_vector_single_plane_line_size_four() {
            let test = TestCase::new(
                // input
                TestTensorParts::new_vector((0..32).collect()).with_line_size(4),
                // output
                TestTensorParts::new_vector(vec![0, 0, 0, 0]).with_line_size(4),
                // expected
                vec![112, 120, 128, 136],
            );
            impl_reduce_sum_test::<TestRuntime, u32>(&Default::default(), test);
        }

        #[test]
        pub fn reduce_sum_lined_vector_single_plane_line_size_four() {
            let mut test = TestCase::new(
                // input
                TestTensorParts::new_vector((0..32).collect()).with_line_size(4),
                // output
                TestTensorParts::new_vector(vec![0]),
                // expected
                vec![496],
            );
            test.reduce_lines = true;
            impl_reduce_sum_test::<TestRuntime, u32>(&Default::default(), test);
        }

        #[test]
        pub fn reduce_sum_vector_long_single_plane() {
            let test = TestCase::new(
                // input
                TestTensorParts::new_vector((0..128).collect()),
                // output
                TestTensorParts::new_vector(vec![0]),
                // expected
                vec![8128],
            );
            impl_reduce_sum_test::<TestRuntime, u32>(&Default::default(), test);
        }

        #[test]
        pub fn reduce_sum_long_vector_four_planes() {
            let mut test = TestCase::new(
                // input
                TestTensorParts::new_vector((0..128).collect()),
                // output
                TestTensorParts::new_vector(vec![0]),
                // expected
                vec![8128],
            );
            test.cube_dim = CubeDim::new(128, 1, 1);
            impl_reduce_sum_test::<TestRuntime, u32>(&Default::default(), test);
        }

        #[test]
        pub fn reduce_sum_vector_with_remainder_single_plane() {
            let test = TestCase::new(
                // input
                TestTensorParts::new_vector((0..128).collect()),
                // output
                TestTensorParts::new_vector(vec![0]),
                // expected
                vec![8128],
            );
            impl_reduce_sum_test::<TestRuntime, u32>(&Default::default(), test);
        }

        #[test]
        pub fn reduce_sum_vector_with_remainder_four_planes() {
            let mut test = TestCase::new(
                // input
                TestTensorParts::new_vector((0..100).collect()),
                // output
                TestTensorParts::new_vector(vec![0]),
                // expected
                vec![4950],
            );
            test.cube_dim = CubeDim::new(128, 1, 1);
            impl_reduce_sum_test::<TestRuntime, u32>(&Default::default(), test);
        }

        #[test]
        pub fn reduce_sum_lined_vector_with_remainder_four_planes() {
            let mut test = TestCase::new(
                // input
                TestTensorParts::new_vector((0..100).collect()).with_line_size(4),
                // output
                TestTensorParts::new_vector(vec![0]),
                // expected
                vec![4950],
            );
            test.cube_dim = CubeDim::new(128, 1, 1);
            test.reduce_lines = true;
            impl_reduce_sum_test::<TestRuntime, u32>(&Default::default(), test);
        }

        #[test]
        pub fn reduce_sum_vector_f32_eight_planes() {
            let mut test = TestCase::new(
                // input
                TestTensorParts::new_vector((0..1024).map(|n| n as f32).collect()),
                // output
                TestTensorParts::new_vector(vec![0.0]),
                // expected
                vec![523776.0],
            );
            test.tolerance = Some(1e-9);
            test.cube_dim = CubeDim::new(256, 1, 1);
            impl_reduce_sum_test::<TestRuntime, f32>(&Default::default(), test);
        }

        #[test]
        pub fn reduce_sum_vector_f32_too_many_planes() {
            let mut test = TestCase::new(
                // input
                TestTensorParts::new_vector((0..128).map(|n| n as f32).collect()),
                // output
                TestTensorParts::new_vector(vec![0.0]),
                // expected
                vec![8128.0],
            );
            test.tolerance = Some(1e-9);
            test.cube_dim = CubeDim::new(256, 1, 1);
            impl_reduce_sum_test::<TestRuntime, f32>(&Default::default(), test);
        }
    };
}

#[derive(Debug)]
pub struct TestCase<N> {
    pub input: TestTensorParts<N>,
    pub output: TestTensorParts<N>,
    pub expected: Vec<N>,
    pub tolerance: Option<f32>,
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
    pub sum_dim: u32,
    pub reduce_lines: bool,
}

impl<N> TestCase<N> {
    pub fn new(input: TestTensorParts<N>, output: TestTensorParts<N>, expected: Vec<N>) -> Self {
        Self {
            input,
            output,
            expected,
            tolerance: None,
            cube_count: CubeCount::Static(1, 1, 1),
            cube_dim: CubeDim::new(32, 1, 1),
            sum_dim: 0,
            reduce_lines: false,
        }
    }
}

#[derive(Debug)]
pub struct TestTensorParts<N> {
    pub values: Vec<N>,
    pub stride: Vec<usize>,
    pub shape: Vec<usize>,
    pub line_size: u8,
}

impl<N> TestTensorParts<N> {
    pub fn new_vector(values: Vec<N>) -> Self {
        let shape = vec![values.len()];
        Self {
            values,
            stride: vec![1],
            shape,
            line_size: 1,
        }
    }

    pub fn with_line_size(mut self, line_size: u8) -> Self {
        self.line_size = line_size;
        self
    }
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
        line_size: test.input.line_size as u32,
        max_num_planes: test.cube_dim.num_elems()
            / client.properties().hardware_properties().plane_size_min,
    };

    unsafe {
        let input_tensor = TensorArg::from_raw_parts::<N>(
            &input_handle,
            &test.input.stride,
            &test.input.shape,
            test.input.line_size,
        );
        let output_tensor = TensorArg::from_raw_parts::<N>(
            &output_handle,
            &test.output.stride,
            &test.output.shape,
            test.output.line_size,
        );

        if test.reduce_lines {
            reduce_sum_lined::launch_unchecked::<N, R>(
                &client,
                test.cube_count,
                test.cube_dim,
                input_tensor,
                output_tensor,
                config,
            );
        } else {
            reduce_sum::launch_unchecked::<N, R>(
                &client,
                test.cube_count,
                test.cube_dim,
                input_tensor,
                output_tensor,
                config,
            );
        }
    }

    let binding = output_handle.binding();
    let bytes = client.read_one(binding);
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
