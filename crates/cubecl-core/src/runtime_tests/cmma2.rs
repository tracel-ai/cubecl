use std::{println, vec, vec::Vec};

use cubecl_ir::{ElemType, FloatKind};

use crate::{self as cubecl, cmma::Cube};
use cubecl::prelude::*;

#[cube(launch)]
/// Executes Out = Lhs @ Rhs.T
pub fn kernel_simple_f16_workgroup_gmem(
    out: &mut [f32],
    beta: f32,
    #[comptime] size: (usize, usize, usize),
) {
    let (m, n, k) = size;

    let matrix = cmma::Matrix::<f32, Cube>::from_slice(
        cmma::MatrixIdent::Accumulator,
        m,
        n,
        k,
        cmma::MatrixLayout::RowMajor,
        &*out,
        n as u32,
    );

    // Use both inputs and capture extra param
    cmma::execute_elementwise_op(&matrix, &matrix, |row, col, elem| {
        elem * row as f32 + col as f32 + beta
    });

    cmma::store(out, &matrix, n as u32, cmma::MatrixLayout::RowMajor);
}

pub fn test_elemwise_cube<R: Runtime>(client: ComputeClient<R>, cube_dimensions: u32) {
    let cd_ty = ElemType::Float(FloatKind::F32).into();
    let config = client.features().matmul.cube_mma.iter().find(|cfg| {
        cfg.cd_type == cd_ty
            && cfg
                .units_per_block
                .map(|it| it == cube_dimensions)
                .unwrap_or(true)
    });
    if config.is_none() {
        // We can't execute the test, skip.
        println!("No valid config for cube matmul, skipping");
        return;
    }

    let config = config.unwrap();
    println!("Running config {config:?} with cube dim {cube_dimensions}");

    let m = config.m_granularity as usize;
    let n = config.n_granularity as usize;
    let k = config.k_granularity as usize;
    let beta = 2.5;

    let acc_size = m * n;

    let data: Vec<f32> = (0..acc_size).map(|_| 1.0).collect();

    let out = client.create_from_slice(f32::as_bytes(&data));

    unsafe {
        kernel_simple_f16_workgroup_gmem::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(cube_dimensions),
            BufferArg::from_raw_parts(out.clone(), acc_size),
            beta,
            (m, n, k),
        )
    };

    let actual = client.read_one_unchecked(out);
    let actual = f32::from_bytes(&actual);

    assert_eq!(test_elementwise_cube_expected(m, n, beta), actual);
}

pub fn test_elementwise_cube_expected(m: usize, n: usize, beta: f32) -> Vec<f32> {
    let acc_size = m * n;

    let mut out = vec![0f32; acc_size];

    for m in 0..m {
        let out_offs = m * n;
        for n in 0..n {
            let value = 1.0 * m as f32 + n as f32 + beta;
            out[out_offs + n] = value;
        }
    }

    out
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma2 {
    () => {
        use super::*;
        use cubecl_core::prelude::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_cmma_elemwise_cube() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::cmma2::test_elemwise_cube::<TestRuntime>(
                client.clone(),
                64,
            );
            cubecl_core::runtime_tests::cmma2::test_elemwise_cube::<TestRuntime>(
                client.clone(),
                128,
            );
            cubecl_core::runtime_tests::cmma2::test_elemwise_cube::<TestRuntime>(client, 256);
        }
    };
}
