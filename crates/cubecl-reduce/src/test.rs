#![allow(missing_docs)]

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::{
    reduce_naive, ReduceArgMax, ReduceArgMin, ReduceNaiveInstruction, ReduceMean, ReduceProd, ReduceSum,
};

#[cube(launch_unchecked)]
pub fn naive_reduce_dim_kernel<I: Numeric, O: Numeric, R: ReduceNaiveInstruction<I>>(
    input: &Tensor<Line<I>>,
    output: &mut Tensor<Line<O>>,
    dim: u32,
) {
    reduce_naive::<R, I, O>(input, output, dim)
}

#[macro_export]
macro_rules! testgen_reduce {
    ([$($float:ident), *]) => {
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
        use cubecl_reduce::test::TestCase;
        use cubecl_core::prelude::CubeCount;

        $crate::impl_test_reduce!(
            $float,
            [
                {
                    id: "reduce_columns_small_matrix_row_major",
                    shape: [4, 8],
                    stride: [8, 1],
                    reduce_dim: 0,
                    cube_count: CubeCount::Static(1, 1, 1),
                    cube_dim: CubeDim::new(4, 8, 1),
                    line_size: 1,
                },
                {
                    id: "reduce_columns_large_matrix_row_major",
                    shape: [8, 256],
                    stride: [256, 1],
                    reduce_dim: 1,
                    cube_count: CubeCount::Static(8, 1, 1),
                    cube_dim: CubeDim::new(16, 16, 1),
                    line_size: 1,
                },
                {
                    id: "reduce_rows_large_matrix_row_major",
                    shape: [8, 256],
                    stride: [256, 1],
                    reduce_dim: 0,
                    cube_count: CubeCount::Static(8, 1, 1),
                    cube_dim: CubeDim::new(16, 16, 1),
                    line_size: 1,
                },
                {
                    id: "rank_three_tensor",
                    shape: [16, 16, 16],
                    stride: [1, 256, 16],
                    reduce_dim: 2,
                    cube_count: CubeCount::Static(4, 1, 1),
                    cube_dim: CubeDim::new(16, 16, 1),
                    line_size: 1,
                },
                {
                    id: "rank_three_tensor_unexact_shape",
                    shape: [11, 12, 13],
                    stride: [156, 13, 1],
                    reduce_dim: 1,
                    cube_count: CubeCount::Static(4, 1, 1),
                    cube_dim: CubeDim::new(16, 16, 1),
                    line_size: 1,
                },
                {
                    id: "reduce_rows_large_matrix_row_major_line_size_four",
                    shape: [32, 64],
                    stride: [64, 1],
                    reduce_dim: 0,
                    cube_count: CubeCount::Static(8, 1, 1),
                    cube_dim: CubeDim::new(16, 16, 1),
                    line_size: 4,
                }
            ]
        );
    };
}

#[macro_export]
macro_rules! impl_test_reduce {
    (
        $float:ident,
        [
            $(
                {
                    id: $id:literal,
                    shape: $shape:expr,
                    stride: $stride:expr,
                    reduce_dim: $reduce_dim:expr,
                    cube_count: $cube_count:expr,
                    cube_dim: $cube_dim:expr,
                    line_size: $line_size:expr,
                }
            ),*
        ]) => {
        ::paste::paste! {
            $(
                #[test]
                pub fn [< reduce_sum_dim_naive_ $id >]() {
                    let test = TestCase {
                        shape: $shape.into(),
                        stride: $stride.into(),
                        reduce_dim: $reduce_dim,
                        cube_count: $cube_count,
                        cube_dim: $cube_dim,
                        line_size:$line_size
                    };
                    test.test_sum_dim_naive::<$float, TestRuntime>(&Default::default());
                }

                #[test]
                pub fn [< reduce_prod_dim_naive_ $id >]() {
                    let test = TestCase {
                        shape: $shape.into(),
                        stride: $stride.into(),
                        reduce_dim: $reduce_dim,
                        cube_count: $cube_count,
                        cube_dim: $cube_dim,
                        line_size:$line_size
                    };
                    test.test_prod_dim_naive::<$float, TestRuntime>(&Default::default());
                }

                #[test]
                pub fn [< reduce_mean_dim_naive_ $id >]() {
                    let test = TestCase {
                        shape: $shape.into(),
                        stride: $stride.into(),
                        reduce_dim: $reduce_dim,
                        cube_count: $cube_count,
                        cube_dim: $cube_dim,
                        line_size:$line_size
                    };
                    test.test_mean_dim_naive::<$float, TestRuntime>(&Default::default());
                }

                // Fix the line issue in argmax before running the test.
                #[ignore]
                #[test]
                pub fn [< reduce_argmax_dim_naive_ $id >]() {
                    let test = TestCase {
                        shape: $shape.into(),
                        stride: $stride.into(),
                        reduce_dim: $reduce_dim,
                        cube_count: $cube_count,
                        cube_dim: $cube_dim,
                        line_size:$line_size
                    };
                    test.test_argmax_dim_naive::<$float, TestRuntime>(&Default::default());
                }

                // Fix the line issue in argmin before running the test.
                #[ignore]
                #[test]
                pub fn [< reduce_argmin_dim_naive_ $id >]() {
                    let test = TestCase {
                        shape: $shape.into(),
                        stride: $stride.into(),
                        reduce_dim: $reduce_dim,
                        cube_count: $cube_count,
                        cube_dim: $cube_dim,
                        line_size:$line_size
                    };
                    test.test_argmin_dim_naive::<$float, TestRuntime>(&Default::default());
                }
            )*
        }
    };
}

#[derive(Debug)]
pub struct TestCase {
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub reduce_dim: u32,
    pub line_size: u8,
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
}

impl TestCase {
    pub fn test_sum_dim_naive<F, R>(&self, device: &R::Device)
    where
        F: Float + CubeElement + std::fmt::Display,
        R: Runtime,
    {
        let input_values: Vec<F> = self.random_input_values();
        let expected_values = self.cpu_sum_dim(&input_values);
        self.run_test::<F, F, R, ReduceSum>(device, input_values, expected_values)
    }

    pub fn test_prod_dim_naive<F, R>(&self, device: &R::Device)
    where
        F: Float + CubeElement + std::fmt::Display,
        R: Runtime,
    {
        let input_values: Vec<F> = self.random_input_values();
        let expected_values = self.cpu_prod_dim(&input_values);
        self.run_test::<F, F, R, ReduceProd>(device, input_values, expected_values)
    }

    pub fn test_mean_dim_naive<F, R>(&self, device: &R::Device)
    where
        F: Float + CubeElement + std::fmt::Display,
        R: Runtime,
    {
        let input_values: Vec<F> = self.random_input_values();
        let expected_values = self.cpu_mean_dim(&input_values);
        self.run_test::<F, F, R, ReduceMean>(device, input_values, expected_values)
    }

    pub fn test_argmax_dim_naive<F, R>(&self, device: &R::Device)
    where
        F: Float + CubeElement + std::fmt::Display,
        R: Runtime,
    {
        let input_values: Vec<F> = self.random_input_values();
        let expected_values = self.cpu_argmax_dim(&input_values);
        self.run_test::<F, u32, R, ReduceArgMax>(device, input_values, expected_values)
    }

    pub fn test_argmin_dim_naive<F, R>(&self, device: &R::Device)
    where
        F: Float + CubeElement + std::fmt::Display,
        R: Runtime,
    {
        let input_values: Vec<F> = self.random_input_values();
        let expected_values = self.cpu_argmin_dim(&input_values);
        self.run_test::<F, u32, R, ReduceArgMin>(device, input_values, expected_values)
    }

    pub fn run_test<I, O, R, K>(
        &self,
        device: &R::Device,
        input_values: Vec<I>,
        expected_values: Vec<O>,
    ) where
        I: Numeric + CubeElement + std::fmt::Display,
        O: Numeric + CubeElement + std::fmt::Display,
        R: Runtime,
        K: ReduceNaiveInstruction<I>,
    {
        let client = R::client(device);

        let input_handle = client.create(I::as_bytes(&input_values));

        let output_handle =
            client.create(O::as_bytes(&vec![O::from_int(0); expected_values.len()]));
        let mut output_shape = self.shape.clone();
        output_shape[self.reduce_dim as usize] = 1;
        let output_stride = self.output_stride();

        unsafe {
            let input_tensor = TensorArg::from_raw_parts::<I>(
                &input_handle,
                &self.stride,
                &self.shape,
                self.line_size,
            );
            let output_tensor = TensorArg::from_raw_parts::<O>(
                &output_handle,
                &output_stride,
                &output_shape,
                self.line_size,
            );

            naive_reduce_dim_kernel::launch_unchecked::<I, O, K, R>(
                &client,
                self.cube_count.clone(),
                self.cube_dim.clone(),
                input_tensor,
                output_tensor,
                ScalarArg::new(self.reduce_dim.clone()),
            );
        }

        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = O::from_bytes(&bytes);

        assert_approx_equal_abs(output_values, &expected_values, 1e-7);
    }

    fn cpu_sum_dim<F: Float>(&self, values: &[F]) -> Vec<F> {
        let mut expected = vec![F::new(0.0); self.num_output_values()];
        for input_index in 0..values.len() {
            let output_index = self.to_output_index(input_index);
            expected[output_index] += values[input_index];
        }
        expected
    }

    fn cpu_prod_dim<F: Float>(&self, values: &[F]) -> Vec<F> {
        let mut expected = vec![F::new(1.0); self.num_output_values()];
        for value_index in 0..values.len() {
            let output_index = self.to_output_index(value_index);
            expected[output_index] *= values[value_index];
        }
        expected
    }

    fn cpu_mean_dim<F: Float>(&self, values: &[F]) -> Vec<F> {
        self.cpu_sum_dim(values)
            .into_iter()
            .map(|sum| sum / F::new(self.shape[self.reduce_dim as usize] as f32))
            .collect()
    }

    fn cpu_argmax_dim<F: Float>(&self, values: &[F]) -> Vec<u32> {
        let mut expected = vec![(F::MIN, 0_u32); self.num_output_values()];
        for input_index in 0..values.len() {
            let output_index = self.to_output_index(input_index);
            let (best, _) = expected[output_index];
            let candidate = values[input_index];
            if candidate > best {
                let coordinate = self.to_input_coordinate(input_index);
                expected[output_index] = (candidate, coordinate[self.reduce_dim as usize] as u32);
            }
        }
        expected.into_iter().map(|(_, i)| i).collect()
    }

    fn cpu_argmin_dim<F: Float>(&self, values: &[F]) -> Vec<u32> {
        let mut expected = vec![(F::MAX, 0_u32); self.num_output_values()];
        for input_index in 0..values.len() {
            let output_index = self.to_output_index(input_index);
            let (best, _) = expected[output_index];
            let candidate = values[input_index];
            if candidate < best {
                let coordinate = self.to_input_coordinate(input_index);
                expected[output_index] = (candidate, coordinate[self.reduce_dim as usize] as u32);
            }
        }
        expected.into_iter().map(|(_, i)| i).collect()
    }

    fn num_output_values(&self) -> usize {
        self.line_size as usize * self.shape.iter().product::<usize>()
            / self.shape[self.reduce_dim as usize]
    }

    fn to_output_index(&self, input_index: usize) -> usize {
        let line_size = self.line_size as usize;
        let mut coordinate = self.to_input_coordinate(input_index / line_size);
        coordinate[self.reduce_dim as usize] = 0;
        self.from_output_coordinate(coordinate) * line_size + input_index % line_size
    }

    fn to_input_coordinate(&self, index: usize) -> Vec<usize> {
        self.stride
            .iter()
            .zip(self.shape.iter())
            .map(|(stride, shape)| (index / stride) % shape)
            .collect()
    }

    fn from_output_coordinate(&self, coordinate: Vec<usize>) -> usize {
        coordinate
            .into_iter()
            .zip(self.output_stride().iter())
            .map(|(c, s)| c * s)
            .sum()
    }

    fn output_stride(&self) -> Vec<usize> {
        let dim_stride = self.stride[self.reduce_dim as usize];
        let dim_shape = self.shape[self.reduce_dim as usize];
        self.stride
            .iter()
            .map(|s| match s.cmp(&dim_stride) {
                std::cmp::Ordering::Equal => 1,
                std::cmp::Ordering::Greater => s / dim_shape,
                std::cmp::Ordering::Less => *s,
            })
            .collect()
    }

    fn random_input_values<F: Float>(&self) -> Vec<F> {
        let size = self.shape.iter().product::<usize>() * self.line_size as usize;

        fn lcg(seed: &mut u64) -> f32 {
            const A: u64 = 1664525;
            const C: u64 = 1013904223;
            const M: f64 = 2u64.pow(32) as f64;

            *seed = (A.wrapping_mul(*seed).wrapping_add(C)) % (1u64 << 32);
            (*seed as f64 / M * 2.0 - 1.0) as f32
        }

        let mut seed = 123456789; // Not really important for testing.
        (0..size).map(|_| F::new(lcg(&mut seed))).collect()
        // (0..size).map(|x| F::new(x as f32)).collect()
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
