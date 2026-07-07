use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_runtime::throughput::{CmmaDims, ComputeCmmaConfig, KernelConfig, ThroughputKey};

use crate::throughput::LaunchConfig;

pub fn build_kernel<R: Runtime>(
    client: &ComputeClient<R>,
    key: ThroughputKey,
    cmma_config: ComputeCmmaConfig,
    config: LaunchConfig,
) -> KernelConfig {
    let client = client.clone();
    let dtype = key.dtype;

    let ops_per_cmma = 2 * cmma_config.cmma_dims.num_elems();
    let out_bytes =
        cmma_config.cmma_dims.m * cmma_config.cmma_dims.n * cmma_config.accumulator_type.size();

    let kernel = Box::new(move |iterations: usize| unsafe {
        let out = client.empty(out_bytes);

        compute_cmma_throughput::launch_unchecked(
            &client,
            CubeCount::Static(config.cube_count as u32, 1, 1),
            CubeDim::new_1d(config.cube_dim as u32),
            config.vector_size,
            BufferArg::from_raw_parts(out, 1),
            iterations,
            cmma_config.cmma_dims,
            dtype.into(),
            cmma_config.accumulator_type.into(),
        )
    });

    let planes_per_cube = config.cube_dim / config.plane_size;
    let ops_count = config.cube_count * planes_per_cube * ops_per_cmma;

    KernelConfig { kernel, ops_count }
}

#[cube(launch_unchecked)]
pub fn compute_cmma_throughput<I: Numeric, ACC: Numeric, N: Size>(
    output: &mut [Vector<I, N>],
    n_iter: usize,
    #[comptime] cmm_dims: CmmaDims,
    #[define(I)] _dtype: StorageType,
    #[define(ACC)] _acc: StorageType,
) {
    let CmmaDims { m, n, k } = cmm_dims;

    let a = cmma::Matrix::<I>::from_value(
        cmma::MatrixIdent::A,
        m,
        n,
        k,
        cmma::MatrixLayout::RowMajor,
        I::cast_from(1),
    );

    let b = cmma::Matrix::<I>::from_value(
        cmma::MatrixIdent::B,
        m,
        n,
        k,
        cmma::MatrixLayout::ColMajor,
        I::cast_from(1),
    );

    let acc = cmma::Matrix::<ACC>::from_value(
        cmma::MatrixIdent::Accumulator,
        m,
        n,
        k,
        cmma::MatrixLayout::Undefined,
        ACC::cast_from(0.0),
    );

    for _ in 0..n_iter {
        cmma::execute(&a, &b, &acc, &acc);
    }

    if ABSOLUTE_POS == 0 {
        cmma::store(output, &acc, n as u32, cmma::MatrixLayout::RowMajor);
    }
}
