//! A kernel that allows no fast math gets each operation rounded on its own, on every backend
//! that can say so, and keeps NaN, infinity and the sign of zero where the device can.

use crate::{self as cubecl};
use cubecl::prelude::*;
use cubecl_ir::features::TypeUsage;
use cubecl_runtime::runtime::Runtime;

/// `(1 + step, 1 + 2 step)` for the smallest power of two `step` whose square is at most half an
/// ulp of one: the square of the first rounds to the second, and fusing it into an add keeps the
/// `step²` that rounding drops.
fn near_one<F: num_traits::Float>() -> (F, F) {
    let mantissa_bits = -F::epsilon().to_f64().unwrap().log2();
    let step = 2f64.powf(-((mantissa_bits + 1.0) / 2.0).ceil());
    (
        F::from(1.0 + step).unwrap(),
        F::from(1.0 + 2.0 * step).unwrap(),
    )
}

#[cube]
fn product_sum<F: Float, N: Size>(
    a: &[Vector<F, N>],
    b: &[Vector<F, N>],
    c: &[Vector<F, N>],
    output: &mut [Vector<F, N>],
) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = a[ABSOLUTE_POS] * b[ABSOLUTE_POS] + c[ABSOLUTE_POS];
    }
}

#[cube(launch_unchecked, fast_math = FastMath::none())]
fn strict_product_sum<F: Float, N: Size>(
    a: &[Vector<F, N>],
    b: &[Vector<F, N>],
    c: &[Vector<F, N>],
    output: &mut [Vector<F, N>],
) {
    product_sum(a, b, c, output);
}

#[cube(fast_math = FastMath::none())]
fn strict_product<F: Float, N: Size>(a: Vector<F, N>, b: Vector<F, N>) -> Vector<F, N> {
    a * b
}

#[cube(launch_unchecked)]
fn strict_product_default_sum<F: Float, N: Size>(
    a: &[Vector<F, N>],
    b: &[Vector<F, N>],
    c: &[Vector<F, N>],
    output: &mut [Vector<F, N>],
) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = strict_product(a[ABSOLUTE_POS], b[ABSOLUTE_POS]) + c[ABSOLUTE_POS];
    }
}

/// Which of the kernels computing `a * b + c` to run.
#[derive(Clone, Copy, Debug)]
pub enum ProductSum {
    /// Allows no fast math.
    Strict,
    /// Multiplies in a function that allows no fast math, and adds in one that says nothing.
    StrictProduct,
}

/// `near_one² - rounded` in each lane of four, through `kernel` at `vectorization`: zero when the
/// product rounds on its own, `step²` when it is fused.
pub fn product_sum_residue<F: Float + num_traits::Float + CubeElement>(
    client: &Client,
    kernel: ProductSum,
    vectorization: usize,
) -> [F; 4] {
    let (near_one, rounded) = near_one::<F>();
    let a = [near_one; 4];
    let c = [-rounded; 4];
    let launch = match kernel {
        ProductSum::Strict => strict_product_sum::launch_unchecked::<F>,
        ProductSum::StrictProduct => strict_product_default_sum::launch_unchecked::<F>,
    };

    let output = client.empty(a.len() * size_of::<F>());
    unsafe {
        launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d((a.len() / vectorization) as u32),
            vectorization,
            BufferArg::from_raw_parts(client.create_from_slice(F::as_bytes(&a)), a.len()),
            BufferArg::from_raw_parts(client.create_from_slice(F::as_bytes(&a)), a.len()),
            BufferArg::from_raw_parts(client.create_from_slice(F::as_bytes(&c)), c.len()),
            BufferArg::from_raw_parts(output.clone(), a.len()),
        )
    };

    let actual = client.read_one_unchecked(output);
    F::from_bytes(&actual).try_into().unwrap()
}

fn assert_strict_rounds_each_operation<
    F: Float + num_traits::Float + CubeElement + core::fmt::Debug,
>(
    client: &Client,
) {
    if !F::supported_uses(client).contains(TypeUsage::Arithmetic) {
        return;
    }
    for kernel in [ProductSum::Strict, ProductSum::StrictProduct] {
        for vectorization in [1usize, 4] {
            assert_eq!(
                product_sum_residue::<F>(client, kernel, vectorization),
                [F::zero(); 4],
                "{kernel:?} fused the {} product into the add at vectorization {vectorization}",
                core::any::type_name::<F>(),
            );
        }
    }
}

/// WGSL cannot forbid one operation's contraction, so its own compiler may still fuse it. Metal
/// has no such call either, and nothing has checked what its compiler does.
fn can_forbid_contraction(client: &Client) -> bool {
    let name = client.name();
    !(name.starts_with("wgpu<wgsl") || name == "wgpu<msl>" || name == "metal")
}

pub fn test_strict_rounds_each_operation<R: Runtime>(client: Client) {
    if !can_forbid_contraction(&client) {
        return;
    }
    assert_strict_rounds_each_operation::<half::f16>(&client);
    assert_strict_rounds_each_operation::<half::bf16>(&client);
    assert_strict_rounds_each_operation::<f32>(&client);
    assert_strict_rounds_each_operation::<f64>(&client);
}

#[cube(launch_unchecked, fast_math = FastMath::none())]
fn strict_dot(a: &[Vector<f32, Const<2>>], b: &[Vector<f32, Const<2>>], output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = Vector::dot(a[ABSOLUTE_POS], b[ABSOLUTE_POS]);
    }
}

/// A dot product is a multiply-add per lane, so it fuses the same way. Each pair cancels the other
/// lane's rounded square exactly, in both lane orders.
pub fn test_strict_dot_rounds_each_operation<R: Runtime>(client: Client) {
    if !can_forbid_contraction(&client) {
        return;
    }
    let (near_one, rounded) = near_one::<f32>();
    let a = [-rounded, near_one, near_one, -rounded];
    let b = [1.0, near_one, near_one, 1.0];
    let output = client.empty(2 * size_of::<f32>());
    unsafe {
        strict_dot::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(2),
            BufferArg::from_raw_parts(client.create_from_slice(f32::as_bytes(&a)), 2),
            BufferArg::from_raw_parts(client.create_from_slice(f32::as_bytes(&b)), 2),
            BufferArg::from_raw_parts(output.clone(), 2),
        )
    };

    let actual = client.read_one_unchecked(output);
    assert_eq!(f32::from_bytes(&actual), [0.0; 2]);
}

#[allow(clippy::eq_op)]
#[cube(launch_unchecked, fast_math = FastMath::none())]
fn strict_identities(input: &[f32], output: &mut [f32]) {
    let infinity = input[0];
    let minus_one = input[1];
    let negative_zero = input[2];
    output[0] = infinity - infinity;
    output[1] = infinity * 0.0;
    output[2] = 0.0 / minus_one;
    output[3] = negative_zero + 0.0;
}

pub fn test_strict_keeps_nan_infinity_and_signed_zero<R: Runtime>(client: Client) {
    if !client.properties().features.ieee_special_values {
        return;
    }
    let input = client.create_from_slice(f32::as_bytes(&[f32::INFINITY, -1.0, -0.0]));
    let output = client.empty(4 * size_of::<f32>());
    unsafe {
        strict_identities::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(input, 3),
            BufferArg::from_raw_parts(output.clone(), 4),
        )
    };

    let actual = client.read_one_unchecked(output);
    let actual = f32::from_bytes(&actual);
    assert!(actual[0].is_nan(), "inf - inf is NaN, got {}", actual[0]);
    assert!(actual[1].is_nan(), "inf * 0 is NaN, got {}", actual[1]);
    assert_eq!(
        actual[2].to_bits(),
        (-0.0f32).to_bits(),
        "0 / -1 is -0, got {}",
        actual[2]
    );
    assert_eq!(
        actual[3].to_bits(),
        0.0f32.to_bits(),
        "-0 + 0 is +0, got {}",
        actual[3]
    );
}

#[allow(clippy::eq_op)]
#[cube(launch_unchecked, fast_math = FastMath::none())]
fn strict_without_arithmetic(input: &[f32], output: &mut [f32]) {
    let nan = input[0];
    let zero = input[1];
    output[0] = select(nan == nan, 1.0f32, 0.0f32);
    output[1] = select(nan.is_nan(), 1.0f32, 0.0f32);
    output[2] = -zero;
}

/// A strict kernel keeps NaN and the sign of zero through comparisons and negation too, which
/// have no arithmetic to fuse and so asked for nothing on their own.
pub fn test_strict_keeps_special_values_without_arithmetic<R: Runtime>(client: Client) {
    if !client.properties().features.ieee_special_values {
        return;
    }
    let input = client.create_from_slice(f32::as_bytes(&[f32::NAN, 0.0]));
    let output = client.empty(3 * size_of::<f32>());
    unsafe {
        strict_without_arithmetic::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(input, 2),
            BufferArg::from_raw_parts(output.clone(), 3),
        )
    };

    let actual = client.read_one_unchecked(output);
    let actual = f32::from_bytes(&actual);
    assert_eq!(actual[0], 0.0, "NaN == NaN is false");
    assert_eq!(actual[1], 1.0, "NaN is NaN");
    assert_eq!(
        actual[2].to_bits(),
        (-0.0f32).to_bits(),
        "-(+0) is -0, got {}",
        actual[2]
    );
}

#[cube(launch_unchecked, fast_math = FastMath::none())]
fn strict_mod_floor(lhs: &[f32], rhs: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = lhs[ABSOLUTE_POS].mod_floor(rhs[ABSOLUTE_POS]);
    }
}

/// `mod_floor` is a multiply-subtract, so it fuses the same way. With `y = 1 + 2⁻¹²` the quotient
/// of `x = 4098.5` floors to `4097`, whose product with `y` is `4098 + 2⁻¹²`: rounded on its own
/// it is `4098`, leaving `0.5`, and fused it leaves `0.5 - 2⁻¹²`.
pub fn test_strict_mod_floor_rounds_each_operation<R: Runtime>(client: Client) {
    if !can_forbid_contraction(&client) {
        return;
    }
    let (near_one, _) = near_one::<f32>();
    let output = client.empty(size_of::<f32>());
    unsafe {
        strict_mod_floor::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(client.create_from_slice(f32::as_bytes(&[4098.5])), 1),
            BufferArg::from_raw_parts(client.create_from_slice(f32::as_bytes(&[near_one])), 1),
            BufferArg::from_raw_parts(output.clone(), 1),
        )
    };

    let actual = client.read_one_unchecked(output);
    assert_eq!(f32::from_bytes(&actual), [0.5]);
}

#[cube(launch_unchecked, fast_math = FastMath::none())]
fn strict_vector_sum_and_dot(input: &[Vector<f32, Const<2>>], output: &mut [f32]) {
    output[0] = input[0].vector_sum();
    output[1] = Vector::dot(input[0], input[1]);
}

/// A lane sum that starts from `+0` turns `-0 + -0` into `+0`.
pub fn test_strict_vector_sum_and_dot_keep_negative_zero<R: Runtime>(client: Client) {
    if !client.properties().features.ieee_special_values {
        return;
    }
    let input = client.create_from_slice(f32::as_bytes(&[-0.0, -0.0, 1.0, 1.0]));
    let output = client.empty(2 * size_of::<f32>());
    unsafe {
        strict_vector_sum_and_dot::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(input, 2),
            BufferArg::from_raw_parts(output.clone(), 2),
        )
    };

    let actual = client.read_one_unchecked(output);
    let actual = f32::from_bytes(&actual);
    assert_eq!(
        actual[0].to_bits(),
        (-0.0f32).to_bits(),
        "vector_sum([-0, -0]) is -0, got {}",
        actual[0]
    );
    assert_eq!(
        actual[1].to_bits(),
        (-0.0f32).to_bits(),
        "dot([-0, -0], [1, 1]) is -0, got {}",
        actual[1]
    );
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_fast_math {
    () => {
        mod fast_math {
            use super::*;

            macro_rules! add_test {
                ($test_name:ident) => {
                    #[$crate::runtime_tests::test_log::test]
                    fn $test_name() {
                        let client = TestRuntime::client(&Default::default());
                        cubecl_core::runtime_tests::fast_math::$test_name::<TestRuntime>(client);
                    }
                };
            }

            add_test!(test_strict_rounds_each_operation);
            add_test!(test_strict_dot_rounds_each_operation);
            add_test!(test_strict_keeps_nan_infinity_and_signed_zero);
            add_test!(test_strict_keeps_special_values_without_arithmetic);
            add_test!(test_strict_mod_floor_rounds_each_operation);
            add_test!(test_strict_vector_sum_and_dot_keep_negative_zero);
        }
    };
}
