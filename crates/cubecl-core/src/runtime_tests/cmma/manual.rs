use crate::{self as cubecl};

use cubecl::prelude::*;

use cubecl_ir::MatrixIdent;
use cubecl_runtime::MmaConfig;

#[cube(launch)]
pub fn kernel_manual<A: Numeric, B: Numeric, CD: Numeric>(
    a: &Tensor<A>,
    b: &Tensor<B>,
    c: &Tensor<CD>,
    out: &mut Tensor<CD>,
    #[comptime] size_m: u32,
    #[comptime] size_n: u32,
    #[comptime] size_k: u32,
) {
    let def = cmma::MmaDefinition::<A, B, CD>::new(size_m, size_n, size_k);
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

    // Load A
    #[unroll]
    for i in 0..line_count_a {
        let mut reg = Line::empty(line_size_a);
        #[unroll]
        for k in 0..line_size_a {
            let n_elem = i * line_size_a + k;
            let (row, col) = def.position_of_nth(lane_id, n_elem, MatrixIdent::A);
            let value = a[row * size_k + col];
            reg[k] = value;
        }
        registers_a.push(reg)
    }

    // Load B
    #[unroll]
    for i in 0..line_count_b {
        let mut reg = Line::empty(line_size_b);
        #[unroll]
        for k in 0..line_size_b {
            let n_elem = i * line_size_b + k;
            let (row, col) = def.position_of_nth(lane_id, n_elem, MatrixIdent::B);
            let value = b[row * size_n + col];
            reg[k] = value;
        }
        registers_b.push(reg)
    }

    // Load C
    #[unroll]
    for i in 0..line_count_c {
        let mut reg = Line::empty(line_size_c);
        #[unroll]
        for k in 0..line_size_c {
            let n_elem = i * line_size_c + k;
            let (row, col) = def.position_of_nth(lane_id, n_elem, MatrixIdent::Accumulator);
            let value = c[row * size_n + col];
            reg[k] = value;
        }
        registers_c.push(reg)
    }

    let registers_d = def.execute(&registers_a, &registers_b, &registers_c);

    // Store D
    #[unroll]
    for i in 0..line_count_d {
        let reg = registers_d[i];
        #[unroll]
        for k in 0..line_size_d {
            let n_elem = i * line_size_d + k;
            let (row, col) = def.position_of_nth(lane_id, n_elem, MatrixIdent::Accumulator);
            out[row * size_n + col] = reg[k];
        }
    }
}

pub fn test_cmma_manual<
    R: Runtime,
    A: CubeElement + Numeric,
    B: CubeElement + Numeric,
    CD: CubeElement + Numeric,
>(
    client: ComputeClient<R::Server>,
    cube_dimensions: CubeDim,
    (m, n, k): (usize, usize, usize),
) {
    if !client.properties().features.mma.contains(&MmaConfig {
        a_type: A::cube_type(),
        b_type: B::cube_type(),
        cd_type: CD::cube_type(),
        m: m as u32,
        n: n as u32,
        k: k as u32,
    }) {
        // We can't execute the test, skip.
        println!(
            "Skipping test for a: {:?} b: {:?}, cd: {:?}, m: {m}, n: {n}, k: {k}",
            A::cube_type(),
            B::cube_type(),
            CD::cube_type()
        );
        return;
    }

    // LHS: matrix where each element = (row_index * 2) + column_index
    let lhs: Vec<A> = (0..m)
        .flat_map(|i| (0..k).map(move |j| A::from_int((i * 2 + j) as i64)))
        .collect();

    // RHS: matrix where each element = (row_index * 3) + column_index
    let rhs: Vec<B> = (0..k)
        .flat_map(|i| (0..n).map(move |j| B::from_int((i * 3 + j) as i64)))
        .collect();

    let lhs = client.create(A::as_bytes(&lhs));
    let rhs = client.create(B::as_bytes(&rhs));
    let out = client.empty(core::mem::size_of::<CD>() * m * n);

    unsafe {
        kernel_manual::launch::<A, B, CD, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dimensions,
            TensorArg::from_raw_parts::<A>(&lhs, &[k, 1], &[m, k], 1),
            TensorArg::from_raw_parts::<B>(&rhs, &[n, 1], &[k, n], 1),
            TensorArg::from_raw_parts::<CD>(&out, &[n, 1], &[m, n], 1),
            TensorArg::from_raw_parts::<CD>(&out, &[n, 1], &[m, n], 1),
            m as u32,
            n as u32,
            k as u32,
        )
    };

    let actual = client.read_one(out);
    let actual = CD::from_bytes(&actual);

    // Calculate expected results (row-major order)
    let mut expected = Vec::with_capacity(m * n);
    for i in 0..m {
        // For each output row
        // For each output row
        for j in 0..n {
            // For each output column
            // For each output column
            let mut sum = 0;
            for l in 0..k {
                // Dot product over k-dimension
                let lhs_val = (i * 2 + l) as i64; // LHS[i, l]
                let rhs_val = (l * 3 + j) as i64; // RHS[l, j]
                sum += lhs_val * rhs_val;
            }
            expected.push(CD::from_int(sum));
        }
    }

    // Need tolerance for slight differences because CPU integer version isn't exactly the same
    // as GPU MMA for fp8. 3% tolerance seems to work for both FP8 types.
    // Existing approximate comparison requires `Float`, so just do a simple one here.
    for (i, (expected_val, actual_val)) in expected.iter().zip(actual).enumerate() {
        let expected_val = expected_val.to_f64().unwrap();
        let actual_val = actual_val.to_f64().unwrap();
        let difference = (expected_val - actual_val).abs();
        let max_difference = expected_val * 0.03;
        if difference > max_difference {
            panic!(
                "Expected != actual at position {i}: (expected: {expected_val}, actual: {actual_val}, difference: {difference}, max_difference: {max_difference})"
            )
        }
    }
}
