#![allow(missing_docs)]

use cubecl_core::{prelude::*, Feature};

use crate::sum::{reduce_sum, reduce_sum_lined, ReduceConfig};

#[macro_export]
macro_rules! impl_test_reduce_sum_vector {
    ($float:ident, [$(($num_values:expr, $cube_size:expr, $line_size:expr)),*]) => {
        ::paste::paste! {
            $(
                #[test]
                pub fn [<reduce_sum_vector_ $num_values _ $cube_size _ $line_size >]() {
                    TestCase::<$float>::sum_vector(32, 32, 1).run::<TestRuntime>(&Default::default());
                }
            )*
        }
    };
}

#[macro_export]
macro_rules! testgen_reduce {
    ([$($float:ident),*]) => {
        mod test_reduce {
            use super::*;
            ::paste::paste! {
                $(mod [<$float _ty>] {
                    use super::*;

                    $crate::testgen_reduce!($float);
                })*
            }
        }
    };

    ($float:ident) => {
        use super::*;
        use cubecl_core::as_type;
        use cubecl_core::prelude::Float;
        use cubecl_core::CubeCount;
        use cubecl_reduce::test::TestCase;

        $crate::impl_test_reduce_sum_vector!(
            $float,
            [
                (32, 32, 1),
                (64, 32, 1),
                (100, 32, 1),
                (1000, 32, 1),
                (2048, 32, 1),
                (32, 64, 1),
                (64, 64, 1),
                (100, 64, 1),
                (1000, 64, 1),
                (2048, 64, 1),
                (32, 1024, 1),
                (64, 1024, 1),
                (100, 1024, 1),
                (1000, 1024, 1),
                (2048, 1024, 1),
                (32, 32, 2),
                (64, 32, 2),
                (100, 32, 2),
                (1000, 32, 2),
                (2048, 32, 2),
                (32, 64, 2),
                (64, 64, 2),
                (100, 64, 2),
                (1000, 64, 2),
                (2048, 64, 2),
                (32, 1024, 2),
                (64, 1024, 2),
                (100, 1024, 2),
                (1000, 1024, 2),
                (2048, 1024, 2),
                (32, 32, 4),
                (64, 32, 4),
                (100, 32, 4),
                (1000, 32, 4),
                (2048, 32, 4),
                (32, 64, 4),
                (64, 64, 4),
                (100, 64, 4),
                (1000, 64, 4),
                (2048, 64, 4),
                (32, 1024, 4),
                (64, 1024, 4),
                (100, 1024, 4),
                (1000, 1024, 4),
                (2048, 1024, 4)
            ]
        );
    };
}

#[derive(Debug)]
pub struct TestTensorParts<N> {
    pub values: Vec<N>,
    pub stride: Vec<usize>,
    pub shape: Vec<usize>,
    pub line_size: u8,
}

impl<N: Float> TestTensorParts<N> {
    pub fn new_vector(values: Vec<N>) -> Self {
        let shape = vec![values.len()];
        Self {
            values,
            stride: vec![1],
            shape,
            line_size: 1,
        }
    }

    pub fn range_vector(stop: usize) -> Self {
        let values = (0..stop).map(|x| N::new(x as f32)).collect();
        Self::new_vector(values)
    }

    pub fn zero_vector(size: usize) -> Self {
        let values = vec![N::new(0.0); size];
        Self::new_vector(values)
    }

    pub fn with_line_size(mut self, line_size: u8) -> Self {
        self.line_size = line_size;
        self
    }
}

#[derive(Debug)]
pub struct TestCase<F> {
    pub input: TestTensorParts<F>,
    pub output: TestTensorParts<F>,
    pub expected: Vec<F>,
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
    pub sum_dim: u32,
    pub reduce_lines: bool,
}

impl<F> TestCase<F> {
    pub fn new(input: TestTensorParts<F>, output: TestTensorParts<F>, expected: Vec<F>) -> Self {
        Self {
            input,
            output,
            expected,
            cube_count: CubeCount::Static(1, 1, 1),
            cube_dim: CubeDim::new(32, 1, 1),
            sum_dim: 0,
            reduce_lines: false,
        }
    }

    /// ASSUMPTION: line_size divide num_values exactly
    pub fn sum_vector(num_values: usize, cube_size: u32, line_size: usize) -> Self
    where
        F: Float,
    {
        // Compute the sums on the cpu.
        let values_per_sum = num_values / line_size;
        let partial_sum = values_per_sum * (values_per_sum - 1) / 2;
        let mut sums = vec![0; line_size];

        #[allow(clippy::needless_range_loop)]
        for k in 0..line_size {
            sums[k] = partial_sum + values_per_sum * k;
        }
        let sums = sums.into_iter().map(|s| F::new(s as f32)).collect();

        let mut test = TestCase::new(
            // input
            TestTensorParts::range_vector(num_values),
            // output
            TestTensorParts::zero_vector(line_size),
            // expected
            sums,
        );
        test.cube_dim = CubeDim::new(cube_size, 1, 1);
        test
    }

    pub fn run<R: Runtime>(self, device: &R::Device)
    where
        F: Float + CubeElement + std::fmt::Display,
    {
        let client = R::client(device);
        if !client.properties().feature_enabled(Feature::Plane) {
            // Can't execute the test.
            return;
        }

        let input_handle = client.create(F::as_bytes(&self.input.values));
        let output_handle = client.create(F::as_bytes(&self.output.values));

        let config = ReduceConfig {
            line_size: self.input.line_size as u32,
            max_num_planes: self.cube_dim.num_elems()
                / client.properties().hardware_properties().plane_size_min,
        };

        unsafe {
            let input_tensor = TensorArg::from_raw_parts::<F>(
                &input_handle,
                &self.input.stride,
                &self.input.shape,
                self.input.line_size,
            );
            let output_tensor = TensorArg::from_raw_parts::<F>(
                &output_handle,
                &self.output.stride,
                &self.output.shape,
                self.output.line_size,
            );

            if self.reduce_lines {
                reduce_sum_lined::launch_unchecked::<F, R>(
                    &client,
                    self.cube_count,
                    self.cube_dim,
                    input_tensor,
                    output_tensor,
                    config,
                );
            } else {
                reduce_sum::launch_unchecked::<F, R>(
                    &client,
                    self.cube_count,
                    self.cube_dim,
                    input_tensor,
                    output_tensor,
                    config,
                );
            }
        }

        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = F::from_bytes(&bytes);

        assert_approx_equal_abs(output_values, &self.expected, 1e-9);
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
