use crate::{self as cubecl, runtime_tests::binary::assert_equals_approx};

use cubecl::prelude::*;

use cubecl_common::{e2m1, e2m1x2, ue8m0};
use cubecl_ir::MatrixIdent;
use cubecl_runtime::ScaledMmaConfig;

#[cube(launch)]
pub fn kernel_scaled<A: CubePrimitive, B: CubePrimitive, CD: Numeric, S: Numeric>(
    a: &Tensor<Line<A>>,
    b: &Tensor<Line<B>>,
    c: &Tensor<Line<CD>>,
    scales_a: &Tensor<S>,
    scales_b: &Tensor<S>,
    out: &mut Tensor<Line<CD>>,
    #[comptime] size_m: u32,
    #[comptime] size_n: u32,
    #[comptime] size_k: u32,
    #[comptime] scales_factor: u32,
) {
    let a_pack = A::packing_factor();
    let b_pack = B::packing_factor();

    let def =
        cmma::MmaDefinition::<A, B, CD>::new_scaled::<S>(size_m, size_n, size_k, scales_factor);
    let lane_id = UNIT_POS_PLANE;

    let elem_count_a = def.elems_per_lane(MatrixIdent::A);
    let line_size_a = def.line_size(MatrixIdent::A);
    let line_count_a = comptime!(elem_count_a / line_size_a);
    let mut registers_a = Sequence::<Line<A>>::new();

    let elem_count_b = def.elems_per_lane(MatrixIdent::B);
    let line_size_b = def.line_size(MatrixIdent::B);
    let line_count_b = comptime!(elem_count_b / line_size_b);
    let mut registers_b = Sequence::<Line<B>>::new();

    let elem_count_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let line_size_c = def.line_size(MatrixIdent::Accumulator);
    let line_count_c = comptime!(elem_count_c / line_size_c);
    let mut registers_c = Sequence::<Line<CD>>::new();

    let elem_count_d = def.elems_per_lane(MatrixIdent::Accumulator);
    let line_size_d = def.line_size(MatrixIdent::Accumulator);
    let line_count_d = comptime!(elem_count_d / line_size_d);

    let scales_count = def.scales_count();
    let mut scales_register_a = Line::<S>::empty(def.scales_line_size());
    let mut scales_register_b = Line::<S>::empty(def.scales_line_size());

    // Load A
    #[unroll]
    for i in 0..line_count_a {
        let n_elem = i * line_size_a * a_pack;
        let (row, col) = def.position_of_nth(lane_id, n_elem, MatrixIdent::A);
        let idx = row * size_k + col;
        let idx = idx / (a.line_size() * a_pack);
        let value = a[idx];

        registers_a.push(value)
    }

    let scales_idx_a = def.scales_index(lane_id, MatrixIdent::A);
    #[unroll]
    for i in 0..scales_count {
        scales_register_a[i] = scales_a[scales_idx_a * scales_factor + i];
    }

    // Load B
    #[unroll]
    for i in 0..line_count_b {
        let n_elem = i * line_size_b * b_pack;
        let (row, col) = def.position_of_nth(lane_id, n_elem, MatrixIdent::B);
        let idx = col * size_k + row;
        let idx = idx / (b.line_size() * b_pack);
        let value = b[idx];

        registers_b.push(value);
    }

    let scales_idx_b = def.scales_index(lane_id, MatrixIdent::B);
    #[unroll]
    for i in 0..scales_count {
        scales_register_b[i] = scales_b[scales_idx_b * scales_factor + i];
    }

    // Load C
    #[unroll]
    for i in 0..line_count_c {
        let n_elem = i * line_size_c;
        let (row, col) = def.position_of_nth(lane_id, n_elem, MatrixIdent::Accumulator);
        let idx = row * size_n + col;
        let value = c[idx / c.line_size()];
        registers_c.push(value)
    }

    let registers_d = def.execute_scaled(
        &registers_a,
        &registers_b,
        &registers_c,
        scales_register_a,
        scales_register_b,
    );

    // Store D
    #[unroll]
    for i in 0..line_count_d {
        let n_elem = i * line_size_d;
        let (row, col) = def.position_of_nth(lane_id, n_elem, MatrixIdent::Accumulator);
        let idx = row * size_n + col;
        out[idx / out.line_size()] = registers_d[i];
    }
}

pub fn test_cmma_scaled<R: Runtime, A: CubeElement + Numeric, B: CubeElement + Numeric>(
    client: ComputeClient<R::Server>,
    cube_dimensions: CubeDim,
    (m, n, k): (usize, usize, usize),
    scales_factor: usize,
) {
    type S = ue8m0;

    let a_elem = A::cube_type();
    let b_elem = B::cube_type();
    let a_line_size = 32 / a_elem.size_bits();
    let b_line_size = 32 / b_elem.size_bits();

    if !client
        .properties()
        .features
        .scaled_mma
        .contains(&ScaledMmaConfig {
            a_type: a_elem,
            b_type: b_elem,
            cd_type: f32::cube_type(),
            scales_type: S::cube_type(),
            m: m as u32,
            n: n as u32,
            k: k as u32,
            scales_factor: scales_factor as u32,
        })
    {
        // We can't execute the test, skip.
        println!(
            "Skipping test for a: {:?}, b: {:?}, scales: {:?} m: {m}, n: {n}, k: {k}",
            A::cube_type(),
            B::cube_type(),
            S::cube_type()
        );
        return;
    }

    // LHS: matrix where each element = (row_index * 2) + column_index
    let lhs: Vec<A> = (0..m)
        .flat_map(|i| (0..k).map(move |j| A::from_int((i * 2 + j) as i64)))
        .collect();
    let lhs_scales: Vec<S> = (0..m)
        .flat_map(|i| (0..scales_factor).map(move |j| S::from_bits((i * 2 + j + 120) as u8)))
        .collect();

    // RHS: matrix where each element = (row_index * 3) + column_index, col-major
    let rhs: Vec<B> = (0..n)
        .flat_map(|j| (0..k).map(move |i| B::from_int((i * 3 + j) as i64)))
        .collect();
    let rhs_scales: Vec<S> = (0..n)
        .flat_map(|j| (0..scales_factor).map(move |i| S::from_bits((i * 3 + j + 120) as u8)))
        .collect();

    let lhs = client.create(A::as_bytes(&lhs));
    let lhs_scales = client.create(S::as_bytes(&lhs_scales));
    let rhs = client.create(B::as_bytes(&rhs));
    let rhs_scales = client.create(S::as_bytes(&rhs_scales));
    let out = client.empty(core::mem::size_of::<f32>() * m * n);

    unsafe {
        kernel_scaled::launch::<A, B, f32, S, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            TensorArg::from_raw_parts::<A>(&lhs, &[k, 1], &[m, k], a_line_size as u8),
            TensorArg::from_raw_parts::<B>(&rhs, &[k, 1], &[n, k], b_line_size as u8),
            TensorArg::from_raw_parts::<f32>(&out, &[n, 1], &[m, n], 1),
            TensorArg::from_raw_parts::<S>(
                &lhs_scales,
                &[scales_factor, 1],
                &[m, scales_factor],
                1,
            ),
            TensorArg::from_raw_parts::<S>(
                &rhs_scales,
                &[scales_factor, 1],
                &[n, scales_factor],
                1,
            ),
            TensorArg::from_raw_parts::<f32>(&out, &[n, 1], &[m, n], 1),
            m as u32,
            n as u32,
            k as u32,
            scales_factor as u32,
        )
    };

    // Calculate expected results (row-major order)
    let mut expected = Vec::with_capacity(m * n);
    for i in 0..m {
        // For each output row
        for j in 0..n {
            // For each output column
            let mut sum = 0.0;
            for l in 0..k {
                let l_scales = l / (k / scales_factor);

                // Dot product over k-dimension
                let lhs_val = (i * 2 + l) as f32; // LHS[i, l]
                let lhs_scale = ue8m0::from_bits((i * 2 + l_scales + 120) as u8).to_f32();
                let rhs_val = (l * 3 + j) as f32; // RHS[l, j]
                let rhs_scale = ue8m0::from_bits((l_scales * 3 + j + 120) as u8).to_f32();
                sum += lhs_val * lhs_scale * rhs_val * rhs_scale;
            }
            expected.push(sum);
        }
    }

    assert_equals_approx::<R, f32>(&client, out, &expected, 0.03);
}

pub fn test_cmma_scaled_fp4<R: Runtime>(
    client: ComputeClient<R::Server>,
    cube_dimensions: CubeDim,
    (m, n, k): (usize, usize, usize),
    scales_factor: usize,
) {
    type AB = e2m1x2;
    type S = ue8m0;

    let ab_elem = AB::cube_type();
    let ab_line_size = 32 / ab_elem.size_bits();

    if !client
        .properties()
        .features
        .scaled_mma
        .contains(&ScaledMmaConfig {
            a_type: ab_elem,
            b_type: ab_elem,
            cd_type: f32::cube_type(),
            scales_type: S::cube_type(),
            m: m as u32,
            n: n as u32,
            k: k as u32,
            scales_factor: scales_factor as u32,
        })
    {
        // We can't execute the test, skip.
        println!(
            "Skipping test for ab: {:?}, scales: {:?} m: {m}, n: {n}, k: {k}",
            AB::cube_type(),
            S::cube_type()
        );
        return;
    }

    // LHS: matrix where each element = (row_index * 2) + column_index
    let lhs_data: Vec<f32> = (0..m)
        .flat_map(|i| (0..k).map(move |j| e2m1::from_bits(((i + j) % 15) as u8 + 1).to_f32()))
        .collect();
    //println!("lhs: {lhs_data:?}");
    let lhs = e2m1x2::from_f32_slice(&lhs_data);
    let lhs_scales_data: Vec<S> = (0..m)
        .flat_map(|i| (0..scales_factor).map(move |j| S::from_bits((i * 2 + j + 120) as u8)))
        .collect();

    // RHS: matrix where each element = (row_index * 3) + column_index, col-major
    let rhs_data: Vec<f32> = (0..n)
        .flat_map(|j| (0..k).map(move |i| e2m1::from_bits(((i + j) % 15) as u8 + 1).to_f32()))
        .collect();
    let rhs = e2m1x2::from_f32_slice(&rhs_data);
    let rhs_scales_data: Vec<S> = (0..n)
        .flat_map(|j| (0..scales_factor).map(move |i| S::from_bits((i * 3 + j + 120) as u8)))
        .collect();

    let lhs = client.create(AB::as_bytes(&lhs));
    let lhs_scales = client.create(S::as_bytes(&lhs_scales_data));
    let rhs = client.create(AB::as_bytes(&rhs));
    let rhs_scales = client.create(S::as_bytes(&rhs_scales_data));
    let out = client.empty(core::mem::size_of::<f32>() * m * n);

    unsafe {
        kernel_scaled::launch::<AB, AB, f32, S, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            TensorArg::from_raw_parts::<AB>(&lhs, &[k / 2, 1], &[m, k / 2], ab_line_size as u8),
            TensorArg::from_raw_parts::<AB>(&rhs, &[k / 2, 1], &[n, k / 2], ab_line_size as u8),
            TensorArg::from_raw_parts::<f32>(&out, &[n, 1], &[m, n], 1),
            TensorArg::from_raw_parts::<S>(
                &lhs_scales,
                &[scales_factor, 1],
                &[m, scales_factor],
                1,
            ),
            TensorArg::from_raw_parts::<S>(
                &rhs_scales,
                &[scales_factor, 1],
                &[n, scales_factor],
                1,
            ),
            TensorArg::from_raw_parts::<f32>(&out, &[n, 1], &[m, n], 1),
            m as u32,
            n as u32,
            k as u32,
            scales_factor as u32,
        )
    };

    // Calculate expected results (row-major order)
    let mut expected = Vec::with_capacity(m * n);
    for i in 0..m {
        // For each output row
        for j in 0..n {
            // For each output column
            let mut sum = 0.0;
            for l in 0..k {
                let l_scales = l / (k / scales_factor);

                // Dot product over k-dimension
                let lhs_val = lhs_data[i * k + l]; // LHS[i, l]
                let lhs_scale = lhs_scales_data[i * scales_factor + l_scales].to_f32();
                let rhs_val = rhs_data[j * k + l];
                let rhs_scale = rhs_scales_data[j * scales_factor + l_scales].to_f32();
                sum += lhs_val * lhs_scale * rhs_val * rhs_scale;
            }
            expected.push(sum);
        }
    }

    assert_equals_approx::<R, f32>(&client, out, &expected, 0.03);
}
