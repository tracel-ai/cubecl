use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::MatmulInstruction;

use crate::matmul::BlockMatmul;

use super::cmma_matmul::BlockInfos;
use super::tile_io::loading::LhsSmemTileReader;
use super::tile_io::loading::RhsSmemTileReader;
use super::tile_io::loading::{LhsTensorLoader, RhsTensorLoader};
use super::tile_io::writing::GmemTensorWriter;
use super::CubeMatmul;
use crate::matmul::tile_io::loading::{new_lhs_tensor_loader, new_rhs_tensor_loader};
use crate::matmul::tile_io::writing::new_tensor_writer;
use crate::matmul::tile_io::TileWriter;

use crate::matmul::cmma_matmul::into_runtime;
use crate::matmul::tests::dummy_tile::array_into_row_major_block_layout;

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
    BM: BlockMatmul<Elem, LhsSmemTileReader<Elem>, RhsSmemTileReader<Elem>, GmemTensorWriter<Elem>>,
    Elem: Numeric,
>(
    lhs_data: Array<Line<Elem>>,
    rhs_data: Array<Line<Elem>>,
    mut out_result: Array<Line<Elem>>,
    #[comptime] layouts: (MatrixLayout, MatrixLayout),
    #[comptime] block_info: BlockInfos,
) {
    let lhs_tile_reader = TileReader;
    let rhs_tile_reader = TileReader;
    let out_writer = GmemTensorWriter;

    let mut acc = BM::acc_init_zeros();
    BM::execute(lhs_tile_reader, rhs_tile_reader, &mut acc, layouts);
    BM::acc_read(&acc, &mut out_writer);

    // let mut lhs_smem = SharedMemory::<Line<E>>::new(BM::M * BM::K);
    // let mut rhs_smem = SharedMemory::<Line<E>>::new(BM::K * BM::N);
    // let out_smem = SharedMemory::<Line<E>>::new(BM::M * BM::N);

    // let lhs_block_info = into_runtime(block_info.lhs);
    // let rhs_block_info = into_runtime(block_info.rhs);
    // let out_block_info = into_runtime(block_info.out);

    // array_into_row_major_block_layout(
    //     lhs_data.as_slice(),
    //     lhs_smem.as_slice_mut(),
    //     lhs_block_info,
    //     false,
    // );
    // array_into_row_major_block_layout(
    //     rhs_data.as_slice(),
    //     rhs_smem.as_slice_mut(),
    //     rhs_block_info,
    //     false,
    // );

    // let lhs = LhsTileReader::<E> {
    //     smem: lhs_smem,
    //     block_info: into_runtime(block_info.lhs),
    // };
    // let rhs = RhsTileReader::<E> {
    //     smem: rhs_smem,
    //     block_info: into_runtime(block_info.rhs),
    // };

    // let mut out = new_out_writer(out_smem, out_block_info);

    // let mut acc = BM::acc_init_zeros();
    // BM::execute(&lhs, &rhs, &mut acc, layouts);
    // BM::acc_read(&mut acc, &mut out);

    // array_into_row_major_block_layout(
    //     out.memory.as_slice(),
    //     out_result.as_slice_mut(),
    //     out_block_info,
    //     true,
    // );
}

#[cube(launch_unchecked)]
pub(crate) fn cube_matmul_launch<
    CM: CubeMatmul<Elem, LhsTensorLoader<Elem>, RhsTensorLoader<Elem>, GmemTensorWriter<Elem>>,
    Elem: Numeric,
>(
    lhs_tensor: Tensor<Line<Elem>>,
    rhs_tensor: Tensor<Line<Elem>>,
    out_tensor: Tensor<Line<Elem>>,
    #[comptime] layouts: (MatrixLayout, MatrixLayout),
    #[comptime] block_info: BlockInfos,
) {
    let k = lhs_tensor.shape(lhs_tensor.rank() - 1);

    let lhs_loader = new_lhs_tensor_loader(lhs_tensor, layouts.0, block_info.lhs);
    let rhs_loader = new_rhs_tensor_loader(rhs_tensor, layouts.1, block_info.rhs);
    let out = new_tensor_writer(out_tensor, block_info.out);

    CM::execute(lhs_loader, rhs_loader, out, (0, k), layouts);
}
