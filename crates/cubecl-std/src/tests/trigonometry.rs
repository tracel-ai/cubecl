use cubecl::prelude::*;
use cubecl_core as cubecl;
use std::f32::consts::{PI, TAU};

use crate::trigonometry::*;

#[cube(launch_unchecked)]
fn kernel_to_degrees(input: &Array<f32>, output: &mut Array<f32>) {
    if UNIT_POS < input.len() {
        output[UNIT_POS] = to_degrees::<f32>(input[UNIT_POS]);
    }
}

pub fn test_to_degrees<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let input_data = vec![0.0, PI / 6.0, PI / 4.0, PI / 2.0, PI, TAU];
    let expected = vec![0.0, 30.0, 45.0, 90.0, 180.0, 360.0];

    let input = client.create(f32::as_bytes(&input_data));
    let output = client.empty(input_data.len() * core::mem::size_of::<f32>());

    unsafe {
        kernel_to_degrees::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(input_data.len() as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input, input_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output, input_data.len(), 1),
        );
    }

    let actual = client.read_one(output);
    let actual = f32::from_bytes(&actual);

    for (i, (&expected_val, &actual_val)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (expected_val - actual_val).abs() < 1e-5,
            "Test {} failed: expected {}, got {}",
            i,
            expected_val,
            actual_val
        );
    }
}

#[cube(launch_unchecked)]
fn kernel_to_radians(input: &Array<f32>, output: &mut Array<f32>) {
    if UNIT_POS < input.len() {
        output[UNIT_POS] = to_radians::<f32>(input[UNIT_POS]);
    }
}

pub fn test_to_radians<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let input_data = vec![0.0, 30.0, 45.0, 90.0, 180.0, 360.0];
    let expected = vec![0.0, PI / 6.0, PI / 4.0, PI / 2.0, PI, TAU];

    let input = client.create(f32::as_bytes(&input_data));
    let output = client.empty(input_data.len() * core::mem::size_of::<f32>());

    unsafe {
        kernel_to_radians::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(input_data.len() as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input, input_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output, input_data.len(), 1),
        );
    }

    let actual = client.read_one(output);
    let actual = f32::from_bytes(&actual);

    for (i, (&expected_val, &actual_val)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (expected_val - actual_val).abs() < 1e-5,
            "Test {} failed: expected {}, got {}",
            i,
            expected_val,
            actual_val
        );
    }
}

#[cube(launch_unchecked)]
fn kernel_sincos(input: &Array<f32>, sin_output: &mut Array<f32>, cos_output: &mut Array<f32>) {
    if UNIT_POS < input.len() {
        let (sin_val, cos_val) = sincos::<f32>(input[UNIT_POS]);
        sin_output[UNIT_POS] = sin_val;
        cos_output[UNIT_POS] = cos_val;
    }
}

pub fn test_sincos<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let input_data = vec![0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0, PI];

    let input = client.create(f32::as_bytes(&input_data));
    let sin_output = client.empty(input_data.len() * core::mem::size_of::<f32>());
    let cos_output = client.empty(input_data.len() * core::mem::size_of::<f32>());

    unsafe {
        kernel_sincos::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(input_data.len() as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input, input_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&sin_output, input_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&cos_output, input_data.len(), 1),
        );
    }

    let actual_sin = client.read_one(sin_output);
    let actual_sin = f32::from_bytes(&actual_sin);
    let actual_cos = client.read_one(cos_output);
    let actual_cos = f32::from_bytes(&actual_cos);

    for (i, &angle) in input_data.iter().enumerate() {
        let expected_sin = angle.sin();
        let expected_cos = angle.cos();

        assert!(
            (expected_sin - actual_sin[i]).abs() < 1e-6,
            "Sin test {} failed: expected {}, got {}",
            i,
            expected_sin,
            actual_sin[i]
        );

        assert!(
            (expected_cos - actual_cos[i]).abs() < 1e-6,
            "Cos test {} failed: expected {}, got {}",
            i,
            expected_cos,
            actual_cos[i]
        );
    }
}

#[cube(launch_unchecked)]
fn kernel_normalize_angle(input: &Array<f32>, output: &mut Array<f32>) {
    if UNIT_POS < input.len() {
        output[UNIT_POS] = normalize_angle::<f32>(input[UNIT_POS]);
    }
}

pub fn test_normalize_angle<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let input_data = vec![
        0.0,
        PI,
        TAU,
        3.0 * PI,
        4.0 * PI,
        -PI,
        -TAU,
        -3.0 * PI,
        PI + 0.5,
        -PI + 0.5,
    ];

    let expected = vec![0.0, PI, 0.0, PI, 0.0, PI, 0.0, PI, PI + 0.5, PI + 0.5];

    let input = client.create(f32::as_bytes(&input_data));
    let output = client.empty(input_data.len() * core::mem::size_of::<f32>());

    unsafe {
        kernel_normalize_angle::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(input_data.len() as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input, input_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output, input_data.len(), 1),
        );
    }

    let actual = client.read_one(output);
    let actual = f32::from_bytes(&actual);

    for (i, (&expected_val, &actual_val)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (expected_val - actual_val).abs() < 1e-5,
            "Test {} failed: expected {}, got {}",
            i,
            expected_val,
            actual_val
        );
    }
}

#[cube(launch_unchecked)]
fn kernel_normalize_angle_signed(input: &Array<f32>, output: &mut Array<f32>) {
    if UNIT_POS < input.len() {
        output[UNIT_POS] = normalize_angle_signed::<f32>(input[UNIT_POS]);
    }
}

pub fn test_normalize_angle_signed<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let input_data = vec![
        0.0,
        PI,
        TAU,
        // 3*PI can result in float errors -> add a small offset to the test
        3.0 * PI + 1e-5,
        4.0 * PI + 1e-5,
        -PI,
        -TAU,
        -3.0 * PI + 1e-5,
        PI + 0.5,
        -PI + 0.5,
    ];

    let expected = vec![
        0.0,
        -PI,
        0.0,
        -PI + 1e-5,
        0.0 + 1e-5,
        -PI,
        0.0,
        -PI + 1e-5,
        -PI + 0.5,
        -PI + 0.5,
    ];

    let input = client.create(f32::as_bytes(&input_data));
    let output = client.empty(input_data.len() * core::mem::size_of::<f32>());

    unsafe {
        kernel_normalize_angle_signed::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(input_data.len() as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input, input_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output, input_data.len(), 1),
        );
    }

    let actual = client.read_one(output);
    let actual = f32::from_bytes(&actual);

    for (i, (&expected_val, &actual_val)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (expected_val - actual_val).abs() < 1e-5,
            "Test {} failed: expected {}, got {}",
            i,
            expected_val,
            actual_val
        );
    }
}

#[cube(launch_unchecked)]
fn kernel_lerp_angle(from: &Array<f32>, to: &Array<f32>, t: &Array<f32>, output: &mut Array<f32>) {
    if UNIT_POS < from.len() {
        output[UNIT_POS] = lerp_angle::<f32>(from[UNIT_POS], to[UNIT_POS], t[UNIT_POS]);
    }
}

pub fn test_lerp_angle<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let from_data = vec![0.0, 0.1, PI - 0.1, 0.0];
    let to_data = vec![PI, TAU - 0.1, PI + 0.1, PI];
    let t_data = vec![0.5, 0.5, 0.5, 0.5];

    let from = client.create(f32::as_bytes(&from_data));
    let to = client.create(f32::as_bytes(&to_data));
    let t = client.create(f32::as_bytes(&t_data));
    let output = client.empty(from_data.len() * core::mem::size_of::<f32>());

    unsafe {
        kernel_lerp_angle::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(from_data.len() as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&from, from_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&to, to_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&t, t_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output, from_data.len(), 1),
        );
    }

    let actual = client.read_one(output);
    let actual = f32::from_bytes(&actual);

    // Test case 0: 0 to π should give π/2
    assert!(
        (actual[0] - PI / 2.0).abs() < 1e-5,
        "Lerp angle test 0 failed"
    );

    // Test case 1: wraparound case - should take shortest path
    assert!(
        actual[1].abs() < 1e-5 || (actual[1] - TAU).abs() < 1e-5,
        "Lerp angle test 1 failed: {}",
        actual[1]
    );

    // Test case 2: small difference around π
    assert!((actual[2] - PI).abs() < 1e-5, "Lerp angle test 2 failed");

    // Test case 3: 0 to π should give π/2
    assert!(
        (actual[3] - PI / 2.0).abs() < 1e-5,
        "Lerp angle test 3 failed"
    );
}

#[cube(launch_unchecked)]
fn kernel_angle_distance(from: &Array<f32>, to: &Array<f32>, output: &mut Array<f32>) {
    if UNIT_POS < from.len() {
        output[UNIT_POS] = angle_distance::<f32>(from[UNIT_POS], to[UNIT_POS]);
    }
}

pub fn test_angle_distance<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let from_data = vec![0.0, 0.1, PI, 0.0];
    let to_data = vec![PI, TAU - 0.1, 0.0, TAU - 0.1];
    let expected = vec![PI, -0.2, -PI, -0.1];

    let from = client.create(f32::as_bytes(&from_data));
    let to = client.create(f32::as_bytes(&to_data));
    let output = client.empty(from_data.len() * core::mem::size_of::<f32>());

    unsafe {
        kernel_angle_distance::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(from_data.len() as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&from, from_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&to, to_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output, from_data.len(), 1),
        );
    }

    let actual = client.read_one(output);
    let actual = f32::from_bytes(&actual);

    for (i, (&expected_val, &actual_val)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (expected_val - actual_val).abs() < 1e-5,
            "Angle distance test {} failed: expected {}, got {}",
            i,
            expected_val,
            actual_val
        );
    }
}

#[cube(launch_unchecked)]
fn kernel_vector_angle_2d(
    x1: &Array<f32>,
    y1: &Array<f32>,
    x2: &Array<f32>,
    y2: &Array<f32>,
    output: &mut Array<f32>,
) {
    if UNIT_POS < x1.len() {
        output[UNIT_POS] =
            vector_angle_2d::<f32>(x1[UNIT_POS], y1[UNIT_POS], x2[UNIT_POS], y2[UNIT_POS]);
    }
}

pub fn test_vector_angle_2d<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    // Simplified test case
    let x1_data = vec![1.0];
    let y1_data = vec![0.0];
    let x2_data = vec![0.0];
    let y2_data = vec![1.0];
    let expected = vec![PI / 2.0];

    let x1 = client.create(f32::as_bytes(&x1_data));
    let y1 = client.create(f32::as_bytes(&y1_data));
    let x2 = client.create(f32::as_bytes(&x2_data));
    let y2 = client.create(f32::as_bytes(&y2_data));
    let output = client.empty(x1_data.len() * core::mem::size_of::<f32>());

    unsafe {
        kernel_vector_angle_2d::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(x1_data.len() as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&x1, x1_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&y1, y1_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&x2, x2_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&y2, y2_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output, x1_data.len(), 1),
        );
    }

    let actual = client.read_one(output);
    let actual = f32::from_bytes(&actual);

    for (i, (&expected_val, &actual_val)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (expected_val - actual_val).abs() < 1e-5,
            "Vector angle 2D test {} failed: expected {}, got {}",
            i,
            expected_val,
            actual_val
        );
    }
}

#[cube(launch_unchecked)]
fn kernel_rotate_2d(
    x: &Array<f32>,
    y: &Array<f32>,
    angle: &Array<f32>,
    x_out: &mut Array<f32>,
    y_out: &mut Array<f32>,
) {
    if UNIT_POS < x.len() {
        let (new_x, new_y) = rotate_2d::<f32>(x[UNIT_POS], y[UNIT_POS], angle[UNIT_POS]);
        x_out[UNIT_POS] = new_x;
        y_out[UNIT_POS] = new_y;
    }
}

pub fn test_rotate_2d<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let x_data = vec![1.0, 0.0, 1.0, 1.0];
    let y_data = vec![0.0, 1.0, 1.0, 0.0];
    let angle_data = vec![PI / 2.0, PI / 2.0, PI / 4.0, PI];

    let expected_x = vec![0.0, -1.0, 0.0, -1.0];
    let expected_y = vec![1.0, 0.0, 1.414213562373095, 0.0];

    let x = client.create(f32::as_bytes(&x_data));
    let y = client.create(f32::as_bytes(&y_data));
    let angle = client.create(f32::as_bytes(&angle_data));
    let x_out = client.empty(x_data.len() * core::mem::size_of::<f32>());
    let y_out = client.empty(y_data.len() * core::mem::size_of::<f32>());

    unsafe {
        kernel_rotate_2d::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(x_data.len() as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&x, x_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&y, y_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&angle, angle_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&x_out, x_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&y_out, y_data.len(), 1),
        );
    }

    let actual_x = client.read_one(x_out);
    let actual_x = f32::from_bytes(&actual_x);
    let actual_y = client.read_one(y_out);
    let actual_y = f32::from_bytes(&actual_y);

    for i in 0..x_data.len() {
        assert!(
            (expected_x[i] - actual_x[i]).abs() < 1e-5,
            "Rotate 2D X test {} failed: expected {}, got {}",
            i,
            expected_x[i],
            actual_x[i]
        );

        assert!(
            (expected_y[i] - actual_y[i]).abs() < 1e-5,
            "Rotate 2D Y test {} failed: expected {}, got {}",
            i,
            expected_y[i],
            actual_y[i]
        );
    }
}

#[cube(launch_unchecked)]
fn kernel_hypot(x: &Array<f32>, y: &Array<f32>, output: &mut Array<f32>) {
    if UNIT_POS < x.len() {
        output[UNIT_POS] = hypot::<f32>(x[UNIT_POS], y[UNIT_POS]);
    }
}

pub fn test_hypot<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let x_data = vec![3.0, 0.0, 1.0, 5.0, 0.0];
    let y_data = vec![4.0, 1.0, 1.0, 12.0, 0.0];
    let expected = vec![5.0, 1.0, 1.4142135623730951, 13.0, 0.0];

    let x = client.create(f32::as_bytes(&x_data));
    let y = client.create(f32::as_bytes(&y_data));
    let output = client.empty(x_data.len() * core::mem::size_of::<f32>());

    unsafe {
        kernel_hypot::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(x_data.len() as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&x, x_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&y, y_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output, x_data.len(), 1),
        );
    }

    let actual = client.read_one(output);
    let actual = f32::from_bytes(&actual);

    for (i, (&expected_val, &actual_val)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (expected_val - actual_val).abs() < 1e-5,
            "Hypot test {} failed: expected {}, got {}",
            i,
            expected_val,
            actual_val
        );
    }
}

#[cube(launch_unchecked)]
fn kernel_sinc(input: &Array<f32>, output: &mut Array<f32>) {
    if UNIT_POS < input.len() {
        output[UNIT_POS] = sinc::<f32>(input[UNIT_POS]);
    }
}

pub fn test_sinc<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let input_data = vec![0.0, 1.0, -1.0, 0.5, -0.5, 2.0];
    // Expected values for normalized sinc function: sin(πx)/(πx)
    let expected = vec![
        1.0,                // sinc(0) = 1
        0.0,                // sinc(1) ≈ 0 (actually 3.8986e-17, but effectively 0)
        0.0,                // sinc(-1) ≈ 0
        0.6366197723675814, // sinc(0.5) = sin(π/2)/(π/2) = 1/(π/2) ≈ 0.6366
        0.6366197723675814, // sinc(-0.5) = sinc(0.5)
        0.0,                // sinc(2) ≈ 0
    ];

    let input = client.create(f32::as_bytes(&input_data));
    let output = client.empty(input_data.len() * core::mem::size_of::<f32>());

    unsafe {
        kernel_sinc::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(input_data.len() as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input, input_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output, input_data.len(), 1),
        );
    }

    let actual = client.read_one(output);
    let actual = f32::from_bytes(&actual);

    for (i, (&expected_val, &actual_val)) in expected.iter().zip(actual.iter()).enumerate() {
        let tolerance = if i == 1 || i == 2 || i == 5 {
            1e-3
        } else {
            1e-5
        }; // More tolerance for near-zero values
        assert!(
            (expected_val - actual_val).abs() < tolerance,
            "Sinc test {} failed: expected {}, got {}",
            i,
            expected_val,
            actual_val
        );
    }
}

#[macro_export]
macro_rules! testgen_trigonometry {
    () => {
        mod trigonometry {
            use super::*;
            use $crate::tests::trigonometry::*;

            #[test]
            fn test_to_degrees_conversion() {
                let client = TestRuntime::client(&Default::default());
                test_to_degrees::<TestRuntime>(client);
            }

            #[test]
            fn test_to_radians_conversion() {
                let client = TestRuntime::client(&Default::default());
                test_to_radians::<TestRuntime>(client);
            }

            #[test]
            fn test_sincos_computation() {
                let client = TestRuntime::client(&Default::default());
                test_sincos::<TestRuntime>(client);
            }

            #[test]
            fn test_normalize_angle_positive() {
                let client = TestRuntime::client(&Default::default());
                test_normalize_angle::<TestRuntime>(client);
            }

            #[test]
            fn test_normalize_angle_signed_range() {
                let client = TestRuntime::client(&Default::default());
                test_normalize_angle_signed::<TestRuntime>(client);
            }

            #[test]
            fn test_lerp_angle_interpolation() {
                let client = TestRuntime::client(&Default::default());
                test_lerp_angle::<TestRuntime>(client);
            }

            #[test]
            fn test_angle_distance_calculation() {
                let client = TestRuntime::client(&Default::default());
                test_angle_distance::<TestRuntime>(client);
            }

            #[test]
            fn test_vector_angle_2d_computation() {
                let client = TestRuntime::client(&Default::default());
                test_vector_angle_2d::<TestRuntime>(client);
            }

            #[test]
            fn test_rotate_2d_transformation() {
                let client = TestRuntime::client(&Default::default());
                test_rotate_2d::<TestRuntime>(client);
            }

            #[test]
            fn test_hypot_computation() {
                let client = TestRuntime::client(&Default::default());
                test_hypot::<TestRuntime>(client);
            }

            #[test]
            fn test_sinc_function() {
                let client = TestRuntime::client(&Default::default());
                test_sinc::<TestRuntime>(client);
            }
        }
    };
}
