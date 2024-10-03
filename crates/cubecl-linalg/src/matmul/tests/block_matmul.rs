use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::BlockKind;
use crate::matmul::BlockMatmul;

use super::dummy_tile::{
    array_into_row_major_block_layout, DummyLhsReader, DummyRhsReader, DummyWriter,
};

#[cube(launch_unchecked)]
fn block_matmul_launch<
    BM: BlockMatmul<E, DummyLhsReader<E>, DummyRhsReader<E>, DummyWriter<E>>,
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

    let lhs = DummyLhsReader::<E> {
        memory: lhs_with_layout,
        block_info: BM::block_info(BlockKind::Lhs),
    };
    let rhs = DummyRhsReader::<E> {
        memory: rhs_with_layout,
        block_info: BM::block_info(BlockKind::Rhs),
    };

    let out_block_info = BM::block_info(BlockKind::Out);
    let mut out = DummyWriter::<E> {
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
