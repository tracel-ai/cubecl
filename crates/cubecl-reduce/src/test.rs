#![allow(missing_docs)]

use cubecl_core::prelude::*;
use rand::{
    distributions::{Distribution, Uniform},
    rngs::StdRng,
    SeedableRng,
};

use crate::{
    reduce, ArgMax, ArgMin, Mean, Prod, ReduceError, ReduceInstruction, ReduceStrategy, Sum,
};

// All random values generated for tests will be in the set
// {-1, -1 + E, -1 + 2E, ..., 1 - E, 1} with E = 1 / PRECISION.
// We choose this set to avoid precision issues with f16 and bf16 and
// also to add multiple similar values to properly test ArgMax and ArgMin.
const PRECISION: i32 = 4;

// This macro generate all the tests.
#[macro_export]
macro_rules! testgen_reduce {
    // Generate all the tests for a list of types.
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

    // Generate all the tests for a specific float type.
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
                    axis: 1,
                },
                {
                    id: "reduce_rows_small_matrix_row_major",
                    shape: [4, 8],
                    stride: [8, 1],
                    axis: 0,
                },
                {
                    id: "reduce_columns_large_matrix_row_major",
                    shape: [8, 256],
                    stride: [256, 1],
                    axis: 1,
                },
                {
                    id: "reduce_rows_large_matrix_row_major",
                    shape: [8, 256],
                    stride: [256, 1],
                    axis: 0,
                },
                {
                    id: "rank_three_tensor",
                    shape: [16, 16, 16],
                    stride: [1, 256, 16],
                    axis: 2,
                },
                {
                    id: "rank_three_tensor_unexact_shape",
                    shape: [11, 12, 13],
                    stride: [156, 13, 1],
                    axis: 1,
                },
                {
                    id: "reduce_rows_large_matrix_row_major_line_size_four",
                    shape: [32, 64],
                    stride: [64, 1],
                    axis: 0,
                }
            ]
        );
    };
}

// For a given tensor description and cube settings
// run the tests for `ReduceSum`, `ReduceProd`, `ReduceMean`, `ReduceArgMax` and `ReduceArgMin`
// for all strategies.
// For each test, a reference reduction is computed on the CPU to compare the outcome of the kernel.
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
                    axis: $axis:expr,
                }
            ),*
        ]
    ) => {
        ::paste::paste! {
            $(
                $crate::impl_test_reduce_with_strategy!{
                    $float,
                    {
                        id: $id,
                        shape: $shape,
                        stride: $stride,
                        axis: $axis,
                    },
                    [ use_planes: false, shared: false ],
                    [ use_planes: true, shared: false ]
                }
            )*
        }
    };
}

#[macro_export]
macro_rules! impl_test_reduce_with_strategy {
    (
        $float:ident,
        {
            id: $id:literal,
            shape: $shape:expr,
            stride: $stride:expr,
            axis: $axis:expr,
        },
        $([use_planes: $use_planes:expr, shared: $shared:expr]),*
    ) => {
        ::paste::paste! {
            $(
                #[test]
                pub fn [< reduce_argmax_ $id _plane_ $use_planes _shared_ $shared >]() {
                    let test = TestCase {
                        shape: $shape.into(),
                        stride: $stride.into(),
                        axis: $axis,
                        strategy: $crate::ReduceStrategy { use_planes: $use_planes, shared: $shared },
                    };
                    test.test_argmax::<$float, TestRuntime>(&Default::default());
                }

                #[test]
                pub fn [< reduce_argmin_ $id _plane_ $use_planes _shared_ $shared >]() {
                    let test = TestCase {
                        shape: $shape.into(),
                        stride: $stride.into(),
                        axis: $axis,
                        strategy: $crate::ReduceStrategy { use_planes: $use_planes, shared: $shared },
                    };
                    test.test_argmin::<$float, TestRuntime>(&Default::default());
                }

                #[test]
                pub fn [< reduce_mean_ $id _plane_ $use_planes _shared_ $shared >]() {
                    let test = TestCase {
                        shape: $shape.into(),
                        stride: $stride.into(),
                        axis: $axis,
                        strategy: $crate::ReduceStrategy { use_planes: $use_planes, shared: $shared },
                    };
                    test.test_mean::<$float, TestRuntime>(&Default::default());
                }

                #[test]
                pub fn [< reduce_prod_ $id _plane_ $use_planes _shared_ $shared >]() {
                    let test = TestCase {
                        shape: $shape.into(),
                        stride: $stride.into(),
                        axis: $axis,
                        strategy: $crate::ReduceStrategy { use_planes: $use_planes, shared: $shared },
                    };
                    test.test_prod::<$float, TestRuntime>(&Default::default());
                }

                #[test]
                pub fn [< reduce_sum_ $id _plane_ $use_planes _shared_ $shared >]() {
                    let test = TestCase {
                        shape: $shape.into(),
                        stride: $stride.into(),
                        axis: $axis,
                        strategy: $crate::ReduceStrategy { use_planes: $use_planes, shared: $shared },
                    };
                    test.test_sum::<$float, TestRuntime>(&Default::default());
                }
            )*
        }
    };
}

#[derive(Debug)]
pub struct TestCase {
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub axis: u32,
    pub strategy: ReduceStrategy,
}

impl TestCase {
    pub fn test_argmax<F, R>(&self, device: &R::Device)
    where
        F: Float + CubeElement + std::fmt::Display,
        R: Runtime,
    {
        let input_values: Vec<F> = self.random_input_values();
        let expected_values = self.cpu_argmax(&input_values);
        self.run_test::<F, u32, R, ArgMax>(device, input_values, expected_values)
    }

    fn cpu_argmax<F: Float>(&self, values: &[F]) -> Vec<u32> {
        let mut expected = vec![(F::MIN, 0_u32); self.num_output_values()];
        for (input_index, &value) in values.iter().enumerate() {
            let output_index = self.to_output_index(input_index);
            let (best, _) = expected[output_index];
            if value > best {
                let coordinate = self.to_input_coordinate(input_index);
                expected[output_index] = (value, coordinate[self.axis as usize] as u32);
            }
        }
        expected.into_iter().map(|(_, i)| i).collect()
    }

    pub fn test_argmin<F, R>(&self, device: &R::Device)
    where
        F: Float + CubeElement + std::fmt::Display,
        R: Runtime,
    {
        let input_values: Vec<F> = self.random_input_values();
        let expected_values = self.cpu_argmin(&input_values);
        self.run_test::<F, u32, R, ArgMin>(device, input_values, expected_values)
    }

    fn cpu_argmin<F: Float>(&self, values: &[F]) -> Vec<u32> {
        let mut expected = vec![(F::MAX, 0_u32); self.num_output_values()];
        for (input_index, &value) in values.iter().enumerate() {
            let output_index = self.to_output_index(input_index);
            let (best, _) = expected[output_index];
            if value < best {
                let coordinate = self.to_input_coordinate(input_index);
                expected[output_index] = (value, coordinate[self.axis as usize] as u32);
            }
        }
        expected.into_iter().map(|(_, i)| i).collect()
    }

    pub fn test_mean<F, R>(&self, device: &R::Device)
    where
        F: Float + CubeElement + std::fmt::Display,
        R: Runtime,
    {
        let input_values: Vec<F> = self.random_input_values();
        let expected_values = self.cpu_mean(&input_values);
        self.run_test::<F, F, R, Mean>(device, input_values, expected_values)
    }

    fn cpu_mean<F: Float>(&self, values: &[F]) -> Vec<F> {
        self.cpu_sum(values)
            .into_iter()
            .map(|sum| sum / F::new(self.shape[self.axis as usize] as f32))
            .collect()
    }

    pub fn test_prod<F, R>(&self, device: &R::Device)
    where
        F: Float + CubeElement + std::fmt::Display,
        R: Runtime,
    {
        let input_values: Vec<F> = self.random_input_values();
        let expected_values = self.cpu_prod(&input_values);
        self.run_test::<F, F, R, Prod>(device, input_values, expected_values)
    }

    fn cpu_prod<F: Float>(&self, values: &[F]) -> Vec<F> {
        let mut expected = vec![F::new(1.0); self.num_output_values()];

        for (input_index, value) in values.iter().enumerate() {
            let output_index = self.to_output_index(input_index);
            expected[output_index] *= *value;
        }
        expected
    }

    pub fn test_sum<F, R>(&self, device: &R::Device)
    where
        F: Float + CubeElement + std::fmt::Display,
        R: Runtime,
    {
        let input_values: Vec<F> = self.random_input_values();
        let expected_values = self.cpu_sum(&input_values);
        self.run_test::<F, F, R, Sum>(device, input_values, expected_values)
    }

    fn cpu_sum<F: Float>(&self, values: &[F]) -> Vec<F> {
        let mut expected = vec![F::new(0.0); self.num_output_values()];

        for (input_index, value) in values.iter().enumerate() {
            let output_index = self.to_output_index(input_index);
            expected[output_index] += *value;
        }
        expected
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
        K: ReduceInstruction<I>,
    {
        let client = R::client(device);

        let input_handle = client.create(I::as_bytes(&input_values));

        // Zero initialize a tensor with the same shape as input
        // except for the `self.axis_red` axis where the shape is 1.
        let output_handle =
            client.create(O::as_bytes(&vec![O::from_int(0); expected_values.len()]));
        let mut output_shape = self.shape.clone();
        output_shape[self.axis as usize] = 1;
        let output_stride = self.output_stride();

        let input = unsafe {
            TensorHandleRef::<R>::from_raw_parts(
                &input_handle,
                &self.stride,
                &self.shape,
                size_of::<I>(),
            )
        };
        let output = unsafe {
            TensorHandleRef::<R>::from_raw_parts(
                &output_handle,
                &output_stride,
                &output_shape,
                size_of::<O>(),
            )
        };

        let result = reduce::<R, I, O, K>(&client, input, output, self.axis, Some(self.strategy));
        if result.is_err_and(|e| e == ReduceError::PlanesUnavailable) {
            return; // We don't test in that case.
        }

        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = O::from_bytes(&bytes);

        assert_approx_equal(output_values, &expected_values);
    }

    fn num_output_values(&self) -> usize {
        self.shape.iter().product::<usize>() / self.shape[self.axis as usize]
    }

    fn to_output_index(&self, input_index: usize) -> usize {
        let mut coordinate = self.to_input_coordinate(input_index);
        coordinate[self.axis as usize] = 0;
        self.from_output_coordinate(coordinate)
    }

    fn to_input_coordinate(&self, index: usize) -> Vec<usize> {
        self.stride
            .iter()
            .zip(self.shape.iter())
            .map(|(stride, shape)| (index / stride) % shape)
            .collect()
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_output_coordinate(&self, coordinate: Vec<usize>) -> usize {
        coordinate
            .into_iter()
            .zip(self.output_stride().iter())
            .map(|(c, s)| c * s)
            .sum()
    }

    fn output_stride(&self) -> Vec<usize> {
        let stride = self.stride[self.axis as usize];
        let shape = self.shape[self.axis as usize];
        self.stride
            .iter()
            .map(|s| match s.cmp(&stride) {
                std::cmp::Ordering::Equal => 1,
                std::cmp::Ordering::Greater => s / shape,
                std::cmp::Ordering::Less => *s,
            })
            .collect()
    }

    fn random_input_values<F: Float>(&self) -> Vec<F> {
        let size = self.shape.iter().product::<usize>();
        let rng = StdRng::seed_from_u64(self.pseudo_random_seed());
        let distribution = Uniform::new_inclusive(-PRECISION, PRECISION);
        let factor = 1.0 / (PRECISION as f32);
        distribution
            .sample_iter(rng)
            .take(size)
            .map(|r| F::new(r as f32 * factor))
            .collect()
    }

    // We don't need a fancy crypto-secure seed as this is only for testing.
    fn pseudo_random_seed(&self) -> u64 {
        123456789
    }

    // fn cpu_prod<F: Float>(&self, values: &[F]) -> Vec<F> {
    //     let mut expected = vec![F::new(1.0); self.num_output_values()];
    //     #[allow(clippy::needless_range_loop)]
    //     for value_index in 0..values.len() {
    //         let output_index = self.to_output_index(value_index);
    //         expected[output_index] *= values[value_index];
    //     }
    //     expected
    // }

    // fn cpu_mean<F: Float>(&self, values: &[F]) -> Vec<F> {
    //     self.cpu_sum(values)
    //         .into_iter()
    //         .map(|sum| sum / F::new(self.shape[self.axis as usize] as f32))
    //         .collect()
    // }

    // fn cpu_argmin<F: Float>(&self, values: &[F]) -> Vec<u32> {
    //     let mut expected = vec![(F::MAX, 0_u32); self.num_output_values()];
    //     #[allow(clippy::needless_range_loop)]
    //     for input_index in 0..values.len() {
    //         let output_index = self.to_output_index(input_index);
    //         let (best, _) = expected[output_index];
    //         let candidate = values[input_index];
    //         if candidate < best {
    //             let coordinate = self.to_input_coordinate(input_index as usize);
    //             expected[output_index] = (candidate, coordinate[self.axis as usize] as u32);
    //         }
    //     }
    //     expected.into_iter().map(|(_, i)| i).collect()
    // }
}

pub fn assert_approx_equal<N: Numeric>(actual: &[N], expected: &[N]) {
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a = a.to_f32().unwrap();
        let e = e.to_f32().unwrap();
        let diff = (a - e).abs();
        if e == 0.0 {
            assert!(
                diff < 1e-10,
                "Values are not approx equal: index={} actual={}, expected={}, difference={}",
                i,
                a,
                e,
                diff,
            );
        } else {
            let rel_diff = diff / e.abs();
            assert!(
                rel_diff < 0.0625,
                "Values are not approx equal: index={} actual={}, expected={}",
                i,
                a,
                e
            );
        }
    }
}
