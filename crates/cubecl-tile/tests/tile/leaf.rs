//! Unit tests for the 1-D register folds in `Monoid::reduce` and `Monoid::reduce_array`.

use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{client::Client, prelude::*, zspace::Shape};
use cubecl_tile::{Monoid, space::UnitShare};

#[cube(launch)]
fn test_hsum_kernel(input: &Tensor<f32>, output: &mut Tensor<f32>) {
    let size!(W4) = 4;
    let size!(W2) = 2;
    let mut v4 = Vector::<f32, W4>::zeroed();
    v4.insert(0usize, input[0]);
    v4.insert(1usize, input[1]);
    v4.insert(2usize, input[2]);
    v4.insert(3usize, input[3]);

    let mut v2 = Vector::<f32, W2>::zeroed();
    v2.insert(0usize, input[0]);
    v2.insert(1usize, input[1]);

    output[0] = Monoid::reduce(v4, 4usize, Monoid::Sum);
    output[1] = Monoid::reduce(v4, 2usize, Monoid::Sum);
    output[2] = Monoid::reduce(v2, 2usize, Monoid::Sum);

    let mut arr = Array::<f32>::new(4usize);
    arr[0] = 1.0f32;
    arr[1] = 2.0f32;
    arr[2] = 3.0f32;
    arr[3] = 4.0f32;
    output[3] = Monoid::reduce_array(
        &arr,
        4usize,
        Monoid::identity::<f32>(Monoid::Sum),
        Monoid::Sum,
    );
    output[4] = Monoid::reduce_array(
        &arr,
        2usize,
        Monoid::identity::<f32>(Monoid::Sum),
        Monoid::Sum,
    );
    output[5] = Monoid::reduce_array(&arr, 4usize, 5.0f32, Monoid::Sum);
}

#[cube(launch)]
fn test_extrema_kernel(input: &Tensor<f32>, output: &mut Tensor<f32>) {
    let size!(W4) = 4;
    let size!(W2) = 2;
    let mut v4 = Vector::<f32, W4>::zeroed();
    v4.insert(0usize, input[0]);
    v4.insert(1usize, input[1]);
    v4.insert(2usize, input[2]);
    v4.insert(3usize, input[3]);

    let mut v2 = Vector::<f32, W2>::zeroed();
    v2.insert(0usize, input[0]);
    v2.insert(1usize, input[1]);

    output[0] = Monoid::reduce(v4, 4usize, Monoid::Max);
    output[1] = Monoid::reduce(v4, 4usize, Monoid::Min);
    output[2] = Monoid::reduce(v2, 2usize, Monoid::Max);
    output[3] = Monoid::reduce(v2, 2usize, Monoid::Min);

    let mut arr = Array::<f32>::new(4usize);
    arr[0] = 3.0f32;
    arr[1] = 1.0f32;
    arr[2] = 7.0f32;
    arr[3] = 2.0f32;
    output[4] = Monoid::reduce_array(
        &arr,
        4usize,
        Monoid::identity::<f32>(Monoid::Max),
        Monoid::Max,
    );
    output[5] = Monoid::reduce_array(
        &arr,
        4usize,
        Monoid::identity::<f32>(Monoid::Min),
        Monoid::Min,
    );
    output[6] = Monoid::reduce_array(&arr, 4usize, 10.0f32, Monoid::Max);
    output[7] = Monoid::reduce_array(&arr, 4usize, -10.0f32, Monoid::Min);
}

#[cube(launch)]
fn test_plane_and_group_kernel(output: &mut Tensor<f32>) {
    let unit_id = UNIT_POS_X;
    let val = (unit_id + 1u32) as f32; // Unit 0: 1.0, Unit 1: 2.0, Unit 2: 3.0, Unit 3: 4.0

    // Non-trivial 4-unit plane operations
    let p_sum = comptime!(UnitShare::Plane).reduce::<f32>(val, Monoid::Sum);
    let p_max = comptime!(UnitShare::Plane).reduce::<f32>(val, Monoid::Max);
    let p_min = comptime!(UnitShare::Plane).reduce::<f32>(val, Monoid::Min);

    // Non-trivial 4-unit butterfly group fold (mask 0b11 = folds all 4 units)
    let size!(W2) = 2;
    let mut v2 = Vector::<f32, W2>::zeroed();
    v2.insert(0usize, val);
    v2.insert(1usize, val * 2.0f32);
    let reduced_full =
        comptime!(UnitShare::Group { unit_bits: 0b11 }).reduce::<Vector<f32, W2>>(v2, Monoid::Sum);

    // 2-unit sub-group butterfly fold (mask 0b01 = folds (0,1) and (2,3) separately)
    let reduced_pair =
        comptime!(UnitShare::Group { unit_bits: 0b01 }).reduce::<Vector<f32, W2>>(v2, Monoid::Sum);

    // The same butterfly under max and min
    let max_full =
        comptime!(UnitShare::Group { unit_bits: 0b11 }).reduce::<Vector<f32, W2>>(v2, Monoid::Max);
    let min_full =
        comptime!(UnitShare::Group { unit_bits: 0b11 }).reduce::<Vector<f32, W2>>(v2, Monoid::Min);
    let min_pair =
        comptime!(UnitShare::Group { unit_bits: 0b01 }).reduce::<Vector<f32, W2>>(v2, Monoid::Min);

    // 1-unit fallback paths (units = 1, mask = 0)
    let s_fallback = comptime!(UnitShare::Repeated).reduce::<f32>(val, Monoid::Sum);
    let m_fallback = comptime!(UnitShare::Repeated).reduce::<f32>(val, Monoid::Max);
    let n_fallback = comptime!(UnitShare::Repeated).reduce::<f32>(val, Monoid::Min);
    let g_fallback = comptime!(UnitShare::Whole).reduce::<Vector<f32, W2>>(v2, Monoid::Sum);

    // Store per-unit results at unit_id * 15
    let base = (unit_id * 15u32) as usize;
    output[base] = p_sum;
    output[base + 1] = p_max;
    output[base + 2] = p_min;
    output[base + 3] = reduced_full.extract(0usize);
    output[base + 4] = reduced_full.extract(1usize);
    output[base + 5] = reduced_pair.extract(0usize);
    output[base + 6] = reduced_pair.extract(1usize);
    output[base + 7] = s_fallback;
    output[base + 8] = m_fallback;
    output[base + 9] = n_fallback;
    output[base + 10] = g_fallback.extract(0usize);
    output[base + 11] = max_full.extract(0usize);
    output[base + 12] = max_full.extract(1usize);
    output[base + 13] = min_full.extract(0usize);
    output[base + 14] = min_pair.extract(0usize);
}

/// CPU planes contain one unit, so only exercise the explicit one-unit fallbacks there.
#[cube(launch)]
fn test_plane_and_group_fallback_kernel(output: &mut Tensor<f32>) {
    let unit_id = UNIT_POS_X;
    let val = (unit_id + 1u32) as f32;
    let size!(W2) = 2;
    let mut v2 = Vector::<f32, W2>::zeroed();
    v2.insert(0usize, val);
    v2.insert(1usize, val * 2.0f32);

    let base = (unit_id * 15u32) as usize;
    output[base + 7] = comptime!(UnitShare::Repeated).reduce::<f32>(val, Monoid::Sum);
    output[base + 8] = comptime!(UnitShare::Repeated).reduce::<f32>(val, Monoid::Max);
    output[base + 9] = comptime!(UnitShare::Repeated).reduce::<f32>(val, Monoid::Min);
    output[base + 10] = comptime!(UnitShare::Whole)
        .reduce::<Vector<f32, W2>>(v2, Monoid::Sum)
        .extract(0usize);
}

#[test]
fn test_hsum_and_array_sum() {
    let client: Client = cubecl::test_device().client();
    let (input_handle, _data) = TestInput::builder(client.clone(), Shape::new([4]))
        .dtype(f32::elem_type_native())
        .custom(vec![1.0, 2.0, 3.0, 4.0])
        .generate_with_f32_host_data();
    let output_handle = TestInput::builder(client.clone(), Shape::new([6]))
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();

    test_hsum_kernel::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        input_handle.binding().into_tensor_arg(),
        output_handle.clone().binding().into_tensor_arg(),
    );

    let output = HostData::from_tensor_handle(&client, output_handle, HostDataType::F32);
    assert_eq!(output.get_f32(&[0]), 10.0); // 1 + 2 + 3 + 4
    assert_eq!(output.get_f32(&[1]), 3.0); // 1 + 2
    assert_eq!(output.get_f32(&[2]), 3.0); // 1 + 2
    assert_eq!(output.get_f32(&[3]), 10.0); // 1 + 2 + 3 + 4 (identity seeded)
    assert_eq!(output.get_f32(&[4]), 3.0); // 1 + 2 (identity seeded)
    assert_eq!(output.get_f32(&[5]), 15.0); // 5 + 1 + 2 + 3 + 4 (seeded)
}

#[test]
fn test_extrema_max_min() {
    let client: Client = cubecl::test_device().client();
    let (input_handle, _data) = TestInput::builder(client.clone(), Shape::new([4]))
        .dtype(f32::elem_type_native())
        .custom(vec![3.0, 1.0, 7.0, 2.0])
        .generate_with_f32_host_data();
    let output_handle = TestInput::builder(client.clone(), Shape::new([8]))
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();

    test_extrema_kernel::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        input_handle.binding().into_tensor_arg(),
        output_handle.clone().binding().into_tensor_arg(),
    );

    let output = HostData::from_tensor_handle(&client, output_handle, HostDataType::F32);
    assert_eq!(output.get_f32(&[0]), 7.0); // max(3, 1, 7, 2)
    assert_eq!(output.get_f32(&[1]), 1.0); // min(3, 1, 7, 2)
    assert_eq!(output.get_f32(&[2]), 3.0); // max(3, 1)
    assert_eq!(output.get_f32(&[3]), 1.0); // min(3, 1)
    assert_eq!(output.get_f32(&[4]), 7.0); // max array (identity seeded with min_value)
    assert_eq!(output.get_f32(&[5]), 1.0); // min array (identity seeded with max_value)
    assert_eq!(output.get_f32(&[6]), 10.0); // max array starting from 10.0
    assert_eq!(output.get_f32(&[7]), -10.0); // min array starting from -10.0
}

#[test]
fn test_plane_and_group_primitives() {
    let client: Client = cubecl::test_device().client();
    let output_handle = TestInput::builder(client.clone(), Shape::new([60]))
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();

    let is_cpu = client.properties().hardware.num_cpu_cores.is_some();
    if is_cpu {
        test_plane_and_group_fallback_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(4),
            output_handle.clone().binding().into_tensor_arg(),
        );
    } else {
        test_plane_and_group_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(4),
            output_handle.clone().binding().into_tensor_arg(),
        );
    }

    let output = HostData::from_tensor_handle(&client, output_handle, HostDataType::F32);

    for plane_unit in 0..4 {
        let base = plane_unit * 15;
        let val = (plane_unit + 1) as f32;

        if !is_cpu {
            // Plane cooperative intrinsics (all units see the whole-plane reduction)
            assert_eq!(
                output.get_f32(&[base]),
                10.0,
                "plane_sum on unit {plane_unit}"
            );
            assert_eq!(
                output.get_f32(&[base + 1]),
                4.0,
                "plane_max on unit {plane_unit}"
            );
            assert_eq!(
                output.get_f32(&[base + 2]),
                1.0,
                "plane_min on unit {plane_unit}"
            );

            // 4-unit butterfly group fold (mask 0b11: all units hold the 4-unit vector total)
            assert_eq!(
                output.get_f32(&[base + 3]),
                10.0,
                "fold_group 0b11 [0] on unit {plane_unit}"
            );
            assert_eq!(
                output.get_f32(&[base + 4]),
                20.0,
                "fold_group 0b11 [1] on unit {plane_unit}"
            );

            // 2-unit pairwise butterfly fold (mask 0b01: units (0,1) and (2,3) fold separately)
            if plane_unit < 2 {
                assert_eq!(
                    output.get_f32(&[base + 5]),
                    3.0,
                    "fold_group 0b01 [0] on unit {plane_unit}"
                );
                assert_eq!(
                    output.get_f32(&[base + 6]),
                    6.0,
                    "fold_group 0b01 [1] on unit {plane_unit}"
                );
            } else {
                assert_eq!(
                    output.get_f32(&[base + 5]),
                    7.0,
                    "fold_group 0b01 [0] on unit {plane_unit}"
                );
                assert_eq!(
                    output.get_f32(&[base + 6]),
                    14.0,
                    "fold_group 0b01 [1] on unit {plane_unit}"
                );
            }
        }

        // 1-unit fallback paths (units = 1, mask = 0)
        assert_eq!(
            output.get_f32(&[base + 7]),
            val,
            "sum 1-unit fallback on unit {plane_unit}"
        );
        assert_eq!(
            output.get_f32(&[base + 8]),
            val,
            "max 1-unit fallback on unit {plane_unit}"
        );
        assert_eq!(
            output.get_f32(&[base + 9]),
            val,
            "min 1-unit fallback on unit {plane_unit}"
        );
        assert_eq!(
            output.get_f32(&[base + 10]),
            val,
            "group mask 0 fallback on unit {plane_unit}"
        );

        if !is_cpu {
            // Max/min butterfly over the same 4 units: vectors are [1,2] [2,4] [3,6] [4,8]
            assert_eq!(
                output.get_f32(&[base + 11]),
                4.0,
                "max_group 0b11 [0] on unit {plane_unit}"
            );
            assert_eq!(
                output.get_f32(&[base + 12]),
                8.0,
                "max_group 0b11 [1] on unit {plane_unit}"
            );
            assert_eq!(
                output.get_f32(&[base + 13]),
                1.0,
                "min_group 0b11 [0] on unit {plane_unit}"
            );
            assert_eq!(
                output.get_f32(&[base + 14]),
                if plane_unit < 2 { 1.0 } else { 3.0 },
                "min_group 0b01 [0] on unit {plane_unit}"
            );
        }
    }
}
