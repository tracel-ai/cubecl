use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::MatmulInstruction;

use crate::matmul::tensor_io::reader::{LhsTensorReader, RhsTensorReader};
use crate::matmul::tensor_io::writer::OutTensorWriter;
use crate::matmul::tests::dummy_tile::array_into_row_major_block_layout;
use crate::matmul::tile_io::reader::{SmemLhsReader, SmemRhsReader};
use crate::matmul::tile_io::writer::DummySmemWriter;
use crate::matmul::BlockKind;
use crate::matmul::BlockMatmul;

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
pub(crate) fn block_matmul_launch<
    BM: BlockMatmul<E, SmemLhsReader<E>, SmemRhsReader<E>, DummySmemWriter<E>>,
    E: Numeric,
>(
    lhs_data: Array<Line<E>>,
    rhs_data: Array<Line<E>>,
    mut out_result: Array<Line<E>>,
    #[comptime] layouts: (MatrixLayout, MatrixLayout),
) {
    let mut lhs_with_layout = SharedMemory::<Line<E>>::new(BM::M * BM::K);
    let mut rhs_with_layout = SharedMemory::<Line<E>>::new(BM::K * BM::N);
    let out_with_layout = SharedMemory::<Line<E>>::new(BM::M * BM::N);

    array_into_row_major_block_layout(
        lhs_data.as_slice(),
        lhs_with_layout.as_slice_mut(),
        BM::block_info(BlockKind::Lhs),
        false,
    );

    array_into_row_major_block_layout(
        rhs_data.as_slice(),
        rhs_with_layout.as_slice_mut(),
        BM::block_info(BlockKind::Rhs),
        false,
    );

    let lhs = SmemLhsReader::<E> {
        memory: lhs_with_layout,
        block_info: BM::block_info(BlockKind::Lhs),
    };
    let rhs = SmemRhsReader::<E> {
        memory: rhs_with_layout,
        block_info: BM::block_info(BlockKind::Rhs),
    };

    let out_block_info = BM::block_info(BlockKind::Out);
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
        BM::block_info(BlockKind::Out),
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
    mut out_tensor: Tensor<Line<E>>,
    #[comptime] layouts: (MatrixLayout, MatrixLayout),
) {
}
