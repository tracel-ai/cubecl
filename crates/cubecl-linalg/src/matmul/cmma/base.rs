use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::block_loop::block_loop;
use super::config::ComptimeCmmaInfo;

#[derive(CubeType, Copy, Clone)]
pub(crate) struct Dimensions {
    pub m: UInt,
    pub k: UInt,
    pub n: UInt,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct Ids {
    pub coop: UInt,
    pub lane: UInt,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct RuntimeCmmaInfo {
    pub ids: Ids,
    pub dims: Dimensions,
    pub offsets: Offsets,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct SharedMemories<FC: Float> {
    pub lhs: SharedMemory<FC>,
    pub rhs: SharedMemory<FC>,
}

#[derive(CubeType, Copy, Clone)]
/// Not divided by vectorization factor
///
/// Note: batch offsets take stride into account, but not the others
pub(crate) struct Offsets {
    pub batch_lhs: UInt,
    pub batch_rhs: UInt,
    pub batch_out: UInt,
    pub cube_row: UInt,
    pub cube_col: UInt,
}

#[derive(CubeType)]
pub(crate) struct CmmaMatrices<F: Float, FC: Float> {
    pub accumulators: Sequence<cmma::Matrix<F>>,
    pub lhs: cmma::Matrix<FC>,
    pub rhs: cmma::Matrix<FC>,
}

#[cube(launch_unchecked)]
#[allow(unused_mut)]
pub fn cmma_kernel<F: Float, FC: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    comptime_info: Comptime<ComptimeCmmaInfo>,
) {
    let ids = get_ids();
    let dims = get_dims::<F>(lhs, rhs);
    let offsets = calculate_offsets::<F>(lhs, rhs, out, comptime_info);
    let runtime_info = RuntimeCmmaInfo { ids, dims, offsets };

    let shared_memories = make_shared_memories::<FC>(comptime_info);
    let cmma_matrices = make_cmma_matrices::<F, FC>(comptime_info);
    block_loop::<F, FC>(
        lhs,
        rhs,
        out,
        shared_memories,
        cmma_matrices,
        runtime_info,
        comptime_info,
    );
}

#[cube]
fn get_dims<F: Float>(lhs: &Tensor<F>, rhs: &Tensor<F>) -> Dimensions {
    let rank = lhs.rank();
    let first_dim = rank - UInt::new(2);
    let second_dim = rank - UInt::new(1);
    let m = lhs.shape(first_dim);
    let k = lhs.shape(second_dim);
    let n = rhs.shape(second_dim);

    Dimensions { m, k, n }
}

#[cube]
fn calculate_offsets<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &Tensor<F>,
    config: Comptime<ComptimeCmmaInfo>,
) -> Offsets {
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let block_size_n = Comptime::map(config, |c| c.block_size_n);

    // Cube offset
    let cube_row = CUBE_POS_X * Comptime::runtime(block_size_m);
    let cube_col = CUBE_POS_Y * Comptime::runtime(block_size_n);

    let rank = out.rank();

    let dim_m = lhs.shape(rank - UInt::new(2));
    let dim_n = rhs.shape(rank - UInt::new(1));

    // Batch offset for output
    let batch_out = dim_m * dim_n * CUBE_POS_Z;
    let mut batch_lhs = UInt::new(0);
    let mut batch_rhs = UInt::new(0);

    // Batch offset for lhs, rhs
    for b in range(0u32, rank - UInt::new(2), Comptime::new(false)) {
        let tmp = batch_out / out.stride(b);
        batch_lhs += tmp % lhs.shape(b) * lhs.stride(b);
        batch_rhs += tmp % rhs.shape(b) * rhs.stride(b);
    }

    Offsets {
        batch_lhs,
        batch_rhs,
        batch_out,
        cube_row,
        cube_col,
    }
}

#[cube]
fn make_shared_memories<FC: Float>(config: Comptime<ComptimeCmmaInfo>) -> SharedMemories<FC> {
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let block_size_n = Comptime::map(config, |c| c.block_size_n);

    let lhs = SharedMemory::<FC>::new(Comptime::get(block_size_k * block_size_m));
    let rhs = SharedMemory::<FC>::new(Comptime::get(block_size_k * block_size_n));

    SharedMemories { lhs, rhs }
}

#[cube]
pub(crate) fn make_cmma_matrices<F: Float, FC: Float>(
    config: Comptime<ComptimeCmmaInfo>,
) -> CmmaMatrices<F, FC> {
    let num_accumulators = Comptime::map(config, |c| c.num_accumulators);
    let mut accumulators = Sequence::<cmma::Matrix<F>>::new();

    for _ in range(0u32, Comptime::get(num_accumulators), Comptime::new(true)) {
        let acc = cmma::Matrix::<F>::new(
            cmma::MatrixIdent::Accumulator,
            16,
            16,
            16,
            cmma::MatrixLayout::Undefined,
        );

        cmma::fill::<F>(&acc, F::new(0.0));

        accumulators.push(acc);
    }

    let lhs = cmma::Matrix::<FC>::new(
        cmma::MatrixIdent::A,
        16,
        16,
        16,
        cmma::MatrixLayout::RowMajor,
    );

    let rhs = cmma::Matrix::<FC>::new(
        cmma::MatrixIdent::B,
        16,
        16,
        16,
        cmma::MatrixLayout::RowMajor,
    );

    CmmaMatrices {
        accumulators,
        lhs,
        rhs,
    }
}

#[cube]
fn get_ids() -> Ids {
    Ids {
        coop: UNIT_POS_Y,
        lane: UNIT_POS_X,
    }
}
