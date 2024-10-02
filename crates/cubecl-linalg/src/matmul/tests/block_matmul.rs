use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_core::CubeElement;

use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::BlockKind;
use crate::matmul::BlockMatmul;

use super::dummy_tile::{
    array_into_row_major_block_layout, DummyLhsReader, DummyRhsReader, DummyWriter,
};
use super::test_utils::assert_equals_approx;
use super::test_utils::matmul_cpu_reference;

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

/// Exported test
pub fn test_block_matmul<BM, E, R>(device: &R::Device)
where
    BM: BlockMatmul<E, DummyLhsReader<E>, DummyRhsReader<E>, DummyWriter<E>>,
    E: Numeric + CubeElement,
    R: Runtime,
{
    let client = R::client(device);
    let lhs_size = (BM::M * BM::K) as usize;
    let rhs_size = (BM::K * BM::N) as usize;
    let out_size = (BM::M * BM::N) as usize;

    let lhs_data: Vec<f32> = (0..lhs_size).map(|x| x as f32 / 100.).collect();
    let rhs_data: Vec<f32> = (0..rhs_size).map(|x| x as f32 / 100.).collect();

    let lhs = client.create(E::as_bytes(&E::from_values(&lhs_data)));
    let rhs = client.create(E::as_bytes(&E::from_values(&rhs_data)));
    let out = client.empty(out_size * E::as_elem().size());

    let cube_dim = BM::resources();
    let cube_count = CubeCount::Static(1, 1, 1);

    unsafe {
        block_matmul_launch::launch_unchecked::<BM, E, R>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(&lhs, lhs_size, 1),
            ArrayArg::from_raw_parts(&rhs, rhs_size, 1),
            ArrayArg::from_raw_parts(&out, out_size, 1),
            (MatrixLayout::Row, MatrixLayout::Row),
        );
    }

    let expected = matmul_cpu_reference(
        &lhs_data,
        &rhs_data,
        BM::M as usize,
        BM::N as usize,
        BM::K as usize,
    );
    if let Err(e) = assert_equals_approx::<E, R>(&client, out, &expected, 10e-1) {
        panic!("{}", e);
    }
}
