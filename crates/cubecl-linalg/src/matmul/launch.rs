use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::MatmulInstruction;

use crate::matmul::tensor_io::reader::{new_lhs_tensor_reader, new_rhs_tensor_reader};
use crate::matmul::tensor_io::reader::{LhsTensorReader, RhsTensorReader};
use crate::matmul::tensor_io::writer::{new_out_writer, OutTensorWriter};
use crate::matmul::tests::dummy_tile::array_into_row_major_block_layout;
use crate::matmul::tile_io::reader::{SmemLhsReader, SmemRhsReader};
use crate::matmul::tile_io::writer::DummySmemWriter;
use crate::matmul::BlockMatmul;

use super::cmma_matmul::{into_runtime, BlockInfos};
use super::CubeMatmul;

#[cube(launch_unchecked)]
pub(crate) fn matmul_instruction_launch<M: MatmulInstruction<I, O>, I: Numeric, O: Numeric>(
    lhs_slice: Array<I>,
    rhs_slice: Array<I>,
    mut out_slice: Array<O>,
    #[comptime] layouts: (MatrixLayout, MatrixLayout),
) {
    let mut lhs = M::init_lhs(layouts.0);
    let mut rhs = M::init_rhs(layouts.1);
    let mut out = M::init_output();

    M::fill_lhs(lhs_slice.as_slice(), &mut lhs);
    M::fill_rhs(rhs_slice.as_slice(), &mut rhs);

    M::execute(&lhs, &rhs, &mut out);
    M::read_output(&out, out_slice.as_slice_mut());
}

#[cube(launch_unchecked)]
/// TODO simplify using smem loading
pub(crate) fn block_matmul_launch<
    BM: BlockMatmul<E, SmemLhsReader<E>, SmemRhsReader<E>, DummySmemWriter<E>>,
    E: Numeric,
>(
    lhs_data: Array<Line<E>>,
    rhs_data: Array<Line<E>>,
    mut out_result: Array<Line<E>>,
    #[comptime] layouts: (MatrixLayout, MatrixLayout),
    #[comptime] block_info: BlockInfos,
) {
    let mut lhs_with_layout = SharedMemory::<Line<E>>::new(BM::M * BM::K);
    let mut rhs_with_layout = SharedMemory::<Line<E>>::new(BM::K * BM::N);
    let out_with_layout = SharedMemory::<Line<E>>::new(BM::M * BM::N);

    let lhs_block_info = into_runtime(block_info.lhs);
    let rhs_block_info = into_runtime(block_info.rhs);
    let out_block_info = into_runtime(block_info.out);

    array_into_row_major_block_layout(
        lhs_data.as_slice(),
        lhs_with_layout.as_slice_mut(),
        lhs_block_info,
        false,
    );

    array_into_row_major_block_layout(
        rhs_data.as_slice(),
        rhs_with_layout.as_slice_mut(),
        rhs_block_info,
        false,
    );

    let lhs = SmemLhsReader::<E> {
        memory: lhs_with_layout,
        block_info: into_runtime(block_info.lhs),
    };
    let rhs = SmemRhsReader::<E> {
        memory: rhs_with_layout,
        block_info: into_runtime(block_info.rhs),
    };

    let mut out = DummySmemWriter::<E> {
        memory: out_with_layout,
        block_info: out_block_info,
    };

    let mut acc = BM::acc_init_zeros();
    BM::execute(lhs, rhs, &mut acc, layouts);
    BM::acc_read(&mut acc, &mut out);

    array_into_row_major_block_layout(
        out.memory.as_slice(),
        out_result.as_slice_mut(),
        out_block_info,
        true,
    );
}

#[cube(launch_unchecked)]
pub(crate) fn cube_matmul_launch<
    CM: CubeMatmul<E, LhsTensorReader<E>, RhsTensorReader<E>, OutTensorWriter<E>>,
    E: Numeric,
>(
    lhs_tensor: Tensor<Line<E>>,
    rhs_tensor: Tensor<Line<E>>,
    out_tensor: Tensor<Line<E>>,
    #[comptime] layouts: (MatrixLayout, MatrixLayout),
    #[comptime] block_info: BlockInfos,
) {
    let k = lhs_tensor.shape(lhs_tensor.rank() - 1);

    let lhs = new_lhs_tensor_reader(lhs_tensor, layouts.0, block_info.lhs);
    let rhs = new_rhs_tensor_reader(rhs_tensor, layouts.1, block_info.rhs);
    let out = new_out_writer(out_tensor, block_info.out);

    CM::execute(lhs, rhs, out, (0, k), layouts);
}
