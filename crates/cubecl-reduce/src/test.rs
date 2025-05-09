#![allow(missing_docs)]

use cubecl_core::prelude::*;
use rand::{
    SeedableRng,
    distr::{Distribution, Uniform},
    rngs::StdRng,
};

use crate::{
    ReduceError, ReduceStrategy, instructions::*, precision::ReducePrecision, reduce, shared_sum,
};

// All random values generated for tests will be in the set
// {-2, -2 + E, -2 + 2E, ..., 2 - E, 2} with E = 1 / PRECISION.
// We choose this set to avoid precision issues with f16 and bf16 and
// also to add multiple similar values to properly test ArgMax and ArgMin.
const PRECISION: i32 = 4;

#[macro_export]
macro_rules! testgen_shared_sum {
    // Generate all the tests for a list of types.
    ([$($float:ident), *]) => {
        mod test_shared_sum {
            use super::*;
            $(
                $crate::testgen_shared_sum!($float);
            )*
        }
    };


    ($float:ident) => {
        ::paste::paste! {
            mod [<$float _ty>] {
                use super::*;

                #[test]
                pub fn vector() {
                    let test = cubecl_reduce::test::TestCase {
                        shape: vec![1024],
                        stride: vec![1],
                        axis: None,
                        strategy: None,
                    };
                    test.test_shared_sum::<$float, TestRuntime>(&Default::default());
                }

                #[test]
                pub fn matrix() {
                    let test = cubecl_reduce::test::TestCase {
                        shape: vec![37, 68],
                        stride: vec![68, 1],
                        axis: None,
                        strategy: None,
                    };
                    test.test_shared_sum::<$float, TestRuntime>(&Default::default());
                }

                #[test]
                pub fn rank_three() {
                    let test = cubecl_reduce::test::TestCase {
                        shape: vec![12, 15, 101],
                        stride: vec![1515, 1, 15],
                        axis: None,
                        strategy: None,
                    };
                    test.test_shared_sum::<$float, TestRuntime>(&Default::default());
                }
            }
        }
    }

}

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

    // Generate all the tests for f32
    () => {
        mod test_reduce {
            use super::*;
            $crate::testgen_reduce!(f32);
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
                    id: "vector_small",
                    shape: [22],
                    stride: [1],
                    axis: 0,
                },
                {
                    id: "vector_large",
                    shape: [1024],
                    stride: [1],
                    axis: 0,
                },
                {
                    id: "parallel_matrix_small",
                    shape: [4, 8],
                    stride: [8, 1],
                    axis: 1,
                },
                {
                    id: "perpendicular_matrix_small",
                    shape: [4, 8],
                    stride: [8, 1],
                    axis: 0,
                },
                {
                    id: "parallel_matrix_large",
                    shape: [8, 256],
                    stride: [256, 1],
                    axis: 1,
                },
                {
                    id: "perpendicular_matrix_large",
                    shape: [8, 256],
                    stride: [256, 1],
                    axis: 0,
                },
                {
                    id: "parallel_rank_three_tensor",
                    shape: [16, 16, 16],
                    stride: [1, 256, 16],
                    axis: 0,
                },
                {
                    id: "perpendicular_rank_three_tensor",
                    shape: [16, 16, 16],
                    stride: [1, 256, 16],
                    axis: 1,
                },
                {
                    id: "perpendicular_rank_three_tensor_unexact_shape",
                    shape: [11, 12, 13],
                    stride: [156, 13, 1],
                    axis: 1,
                },
                {
                    id: "parallel_rank_three_tensor_unexact_shape",
                    shape: [11, 12, 13],
                    stride: [156, 13, 1],
                    axis: 2,
                },
                {
                    id: "parallel_rank_four_tensor",
                    shape: [4, 4, 4, 4],
                    stride: [1, 16, 64, 4],
                    axis: 0,
                },
                {
                    id: "perpendicular_rank_four_tensor",
                    shape: [4, 4, 4, 4],
                    stride: [1, 16, 64, 4],
                    axis: 1,
                },
                {
                    id: "decreasing_rank_four_tensor",
                    shape: [4, 4, 4, 4],
                    stride: [64, 16, 4, 1],
                    axis: 3,
                },
                {
                    id: "parallel_matrix_with_jumps",
                    shape: [8, 8],
                    stride: [64, 1],
                    axis: 1,
                },
                {
                    id: "perpendicular_matrix_with_jumps",
                    shape: [8, 8],
                    stride: [64, 1],
                    axis: 0,
                },
                {
                    id: "broadcast_slice_0",
                    shape: [4, 32],
                    stride: [0, 1],
                    axis: 0,
                }
            ]
        );
    };
}

// For a given tensor description and cube settings
// run the tests for `Sum`, `Prod`, `Mean`, `ArgMax` and `ArgMin`
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
                    [ use_planes: true, shared: false ],
                    [ use_planes: false, shared: true ],
                    [ use_planes: true, shared: true ]
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
                pub fn [< argmax_plane_ $use_planes _shared_ $shared _ $id >]() {
                    let test = TestCase {
                        shape: $shape.into(),
                        stride: $stride.into(),
                        axis: Some($axis),
                        strategy: Some($crate::ReduceStrategy { use_planes: $use_planes, shared: $shared }),
                    };
                    test.test_argmax::<$float, TestRuntime>(&Default::default());
                }


                #[test]
                pub fn [< argmin_plane_ $use_planes _shared_ $shared _ $id >]() {
                    let test = TestCase {
                        shape: $shape.into(),
                        stride: $stride.into(),
                        axis: Some($axis),
                        strategy: Some($crate::ReduceStrategy { use_planes: $use_planes, shared: $shared }),
                    };
                    test.test_argmin::<$float, TestRuntime>(&Default::default());
                }

                #[test]
                pub fn [< mean_plane_ $use_planes _shared_ $shared _ $id >]() {
                    let test = TestCase {
                        shape: $shape.into(),
                        stride: $stride.into(),
                        axis: Some($axis),
                        strategy: Some($crate::ReduceStrategy { use_planes: $use_planes, shared: $shared }),
                    };
                    test.test_mean::<$float, TestRuntime>(&Default::default());
                }

                #[test]
                pub fn [< prod_plane_ $use_planes _shared_ $shared _ $id >]() {
                    let test = TestCase {
                        shape: $shape.into(),
                        stride: $stride.into(),
                        axis: Some($axis),
                        strategy: Some($crate::ReduceStrategy { use_planes: $use_planes, shared: $shared }),
                    };
                    test.test_prod::<$float, TestRuntime>(&Default::default());
                }

                #[test]
                pub fn [< sum_plane_ $use_planes _shared_ $shared _ $id >]() {
                    let test = TestCase {
                        shape: $shape.into(),
                        stride: $stride.into(),
                        axis: Some($axis),
                        strategy: Some($crate::ReduceStrategy { use_planes: $use_planes, shared: $shared }),
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
    pub axis: Option<usize>,
    pub strategy: Option<ReduceStrategy>,
}

impl TestCase {
    pub fn test_argmax<F, R>(&self, device: &R::Device)
    where
        F: ReducePrecision + std::fmt::Display,
        F::EI: CubeElement + Float,
        R: Runtime,
    {
        let input_values: Vec<F::EI> = self.random_input_values();
        let expected_values = match self.axis {
            Some(axis) if self.stride[axis] == 0 => vec![0; input_values.len()],
            _ => self.cpu_argmax(&input_values),
        };
        self.run_reduce_test::<F, u32, R, ArgMax>(device, input_values, expected_values)
    }

    fn cpu_argmax<F: Float>(&self, values: &[F]) -> Vec<u32> {
        let mut expected = vec![(F::min_value(), 0_u32); self.num_output_values()];
        for (input_index, &value) in values.iter().enumerate() {
            if let Some(output_index) = self.to_output_index(input_index) {
                let (best, _) = expected[output_index];
                if value > best {
                    let coordinate = self.to_input_coordinate(input_index).unwrap();
                    expected[output_index] = (value, coordinate[self.axis.unwrap()] as u32);
                }
            }
        }
        expected.into_iter().map(|(_, i)| i).collect()
    }

    pub fn test_argmin<F, R>(&self, device: &R::Device)
    where
        F: ReducePrecision + std::fmt::Display,
        F::EI: CubeElement + Float,
        R: Runtime,
    {
        let input_values: Vec<F::EI> = self.random_input_values();
        let expected_values = match self.axis {
            Some(axis) if self.stride[axis] == 0 => vec![0; input_values.len()],
            _ => self.cpu_argmin(&input_values),
        };
        self.run_reduce_test::<F, u32, R, ArgMin>(device, input_values, expected_values)
    }

    fn cpu_argmin<F: Float>(&self, values: &[F]) -> Vec<u32> {
        let mut expected = vec![(F::max_value(), 0_u32); self.num_output_values()];
        for (input_index, &value) in values.iter().enumerate() {
            if let Some(output_index) = self.to_output_index(input_index) {
                let (best, _) = expected[output_index];
                if value < best {
                    let coordinate = self.to_input_coordinate(input_index).unwrap();
                    expected[output_index] = (value, coordinate[self.axis.unwrap()] as u32);
                }
            }
        }
        expected.into_iter().map(|(_, i)| i).collect()
    }

    pub fn test_mean<F, R>(&self, device: &R::Device)
    where
        F: ReducePrecision + std::fmt::Display,
        F::EI: CubeElement + Float + std::fmt::Display,
        R: Runtime,
    {
        let input_values: Vec<F::EI> = self.random_input_values();
        let expected_values = match self.axis {
            Some(axis) if self.stride[axis] == 0 => input_values.clone(),
            _ => self.cpu_mean(&input_values),
        };
        self.run_reduce_test::<F, F::EI, R, Mean>(device, input_values, expected_values)
    }

    fn cpu_mean<F: Float>(&self, values: &[F]) -> Vec<F> {
        self.cpu_sum(values)
            .into_iter()
            .map(|sum| sum / F::new(self.shape[self.axis.unwrap()] as f32))
            .collect()
    }

    pub fn test_prod<F, R>(&self, device: &R::Device)
    where
        F: ReducePrecision + std::fmt::Display,
        F::EI: CubeElement + Float + std::fmt::Display,
        R: Runtime,
    {
        let input_values: Vec<F::EI> = self.random_input_values();
        let expected_values = match self.axis {
            Some(axis) if self.stride[axis] == 0 => input_values
                .iter()
                .map(|v| Self::powf(*v, self.shape[axis]))
                .collect(),
            _ => self.cpu_prod(&input_values),
        };
        self.run_reduce_test::<F, F::EI, R, Prod>(device, input_values, expected_values)
    }

    fn powf<F: Float>(base: F, power: usize) -> F {
        let mut result = F::new(1.0);
        for _ in 0..power {
            result *= base;
        }
        result
    }

    fn cpu_prod<F: Float>(&self, values: &[F]) -> Vec<F> {
        let mut expected = vec![F::new(1.0); self.num_output_values()];

        for (input_index, value) in values.iter().enumerate() {
            if let Some(output_index) = self.to_output_index(input_index) {
                expected[output_index] *= *value;
            }
        }
        expected
    }

    pub fn test_sum<F, R>(&self, device: &R::Device)
    where
        F: ReducePrecision + std::fmt::Display,
        F::EI: CubeElement + Float + std::fmt::Display,
        R: Runtime,
    {
        let input_values: Vec<F::EI> = self.random_input_values();
        let expected_values = match self.axis {
            Some(axis) if self.stride[axis] == 0 => input_values
                .iter()
                .map(|v| *v * F::EI::from_int(self.shape[axis] as i64))
                .collect(),
            _ => self.cpu_sum(&input_values),
        };
        self.run_reduce_test::<F, F::EI, R, Sum>(device, input_values, expected_values)
    }

    fn cpu_sum<F: Float>(&self, values: &[F]) -> Vec<F> {
        let mut expected = vec![F::new(0.0); self.num_output_values()];

        for (input_index, value) in values.iter().enumerate() {
            if let Some(output_index) = self.to_output_index(input_index) {
                expected[output_index] += *value;
            }
        }
        expected
    }

    pub fn test_shared_sum<F, R>(&self, device: &R::Device)
    where
        F: Float + CubeElement + std::fmt::Display,
        R: Runtime,
    {
        let input_values: Vec<F> = self.random_input_values();
        let mut expected = F::from_int(0);
        for v in input_values.iter() {
            expected += *v;
        }
        self.run_shared_sum_test::<F, R>(device, input_values, expected);
    }

    pub fn run_reduce_test<P, O, R, K>(
        &self,
        device: &R::Device,
        input_values: Vec<P::EI>,
        expected_values: Vec<O>,
    ) where
        P: ReducePrecision,
        P::EI: CubeElement,
        O: Numeric + CubeElement + std::fmt::Display,
        R: Runtime,
        K: ReduceFamily<Config = ()>,
    {
        let client = R::client(device);

        let input_handle = client.create(<P::EI as CubeElement>::as_bytes(&input_values));

        // Zero initialize a tensor with the same shape as input
        // except for the `self.axis` axis where the shape is 1.
        let output_handle =
            client.create(O::as_bytes(&vec![O::from_int(0); expected_values.len()]));
        let mut output_shape = self.shape.clone();
        output_shape[self.axis.unwrap()] = 1;
        let output_stride = self.output_stride();

        let input = unsafe {
            TensorHandleRef::<R>::from_raw_parts(
                &input_handle,
                &self.stride,
                &self.shape,
                size_of::<P>(),
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

        let result = reduce::<R, P, O, K>(
            &client,
            input,
            output,
            self.axis.unwrap(),
            self.strategy,
            (),
        );
        if result.is_err_and(|e| {
            e == ReduceError::PlanesUnavailable || e == ReduceError::ImprecisePlaneDim
        }) {
            return; // We don't test in that case.
        }

        let binding = output_handle.binding();
        let bytes = client.read_one(binding);
        let output_values = O::from_bytes(&bytes);
        assert_approx_equal(output_values, &expected_values);
    }

    pub fn run_shared_sum_test<F, R>(&self, device: &R::Device, input_values: Vec<F>, expected: F)
    where
        F: Float + CubeElement + std::fmt::Display,
        R: Runtime,
    {
        let client = R::client(device);

        let input_handle = client.create(F::as_bytes(&input_values));
        let output_handle = client.create(F::as_bytes(&[F::from_int(0)]));

        let input = unsafe {
            TensorHandleRef::<R>::from_raw_parts(
                &input_handle,
                &self.stride,
                &self.shape,
                size_of::<F>(),
            )
        };
        let output = unsafe {
            TensorHandleRef::<R>::from_raw_parts(&output_handle, &[1], &[1], size_of::<F>())
        };

        let cube_count = 3;
        let result = shared_sum::<R, F>(&client, input, output, cube_count);

        if result.is_err() {
            return; // don't execute the test in that case since atomic adds are not supported.
        }
        let bytes = client.read_one(output_handle.binding());
        let actual = F::from_bytes(&bytes);
        assert_approx_equal(actual, &[expected]);
    }

    fn num_output_values(&self) -> usize {
        self.shape.iter().product::<usize>() / self.shape[self.axis.unwrap()]
    }

    fn to_output_index(&self, input_index: usize) -> Option<usize> {
        let mut coordinate = self.to_input_coordinate(input_index)?;
        coordinate[self.axis.unwrap()] = 0;
        Some(self.from_output_coordinate(coordinate))
    }

    fn to_input_coordinate(&self, index: usize) -> Option<Vec<usize>> {
        let coordinate = self
            .stride
            .iter()
            .zip(self.shape.iter())
            .map(|(stride, shape)| {
                if *stride > 0 {
                    (index / stride) % shape
                } else {
                    index % shape
                }
            })
            .collect::<Vec<usize>>();
        self.validate_input_index(index, &coordinate)
            .then_some(coordinate)
    }

    fn validate_input_index(&self, index: usize, coordinate: &[usize]) -> bool {
        coordinate
            .iter()
            .zip(self.stride.iter())
            .map(|(c, s)| c * s)
            .sum::<usize>()
            == index
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
        self.shape
            .iter()
            .enumerate()
            .scan(1, |stride, (axis, shape)| {
                if axis == self.axis.unwrap() {
                    Some(1)
                } else {
                    let current = Some(*stride);
                    *stride *= shape;
                    current
                }
            })
            .collect()
    }

    fn random_input_values<F: Float>(&self) -> Vec<F> {
        let size = self.input_size();
        let rng = StdRng::seed_from_u64(self.pseudo_random_seed());
        let distribution = Uniform::new_inclusive(-2 * PRECISION, 2 * PRECISION).unwrap();
        let factor = 1.0 / (PRECISION as f32);
        distribution
            .sample_iter(rng)
            .take(size)
            .map(|r| F::new(r as f32 * factor))
            .collect()
        // (0..size).map(|x| F::from_int(x as i64)).collect() TODO DELETE
    }

    fn input_size(&self) -> usize {
        let (stride, shape) = self
            .stride
            .iter()
            .zip(self.shape.iter())
            .max_by_key(|(stride, _)| *stride)
            .unwrap();
        stride * shape
    }

    // We don't need a fancy crypto-secure seed as this is only for testing.
    fn pseudo_random_seed(&self) -> u64 {
        123456789
    }
}

pub fn assert_approx_equal<N: Numeric>(actual: &[N], expected: &[N]) {
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a = a.to_f32().unwrap();
        let e = e.to_f32().unwrap();
        let diff = (a - e).abs();
        if e == 0.0 {
            assert!(
                diff < 1e-10,
                "Values are not approx equal: index={i} actual={a}, expected={e}, difference={diff}",
            );
        } else {
            let rel_diff = diff / e.abs();
            assert!(
                rel_diff < 0.0625,
                "Values are not approx equal: index={i} actual={a}, expected={e}"
            );
        }
    }
}
