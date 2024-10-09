use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::tile_io::Loader;
use crate::matmul::MatmulInstruction;

use crate::matmul::block_info::BlockInfos;
use crate::matmul::BlockMatmul;

use super::tile_io::loading::LhsSmemTileReader;
use super::tile_io::loading::RhsSmemTileReader;
use super::tile_io::writing::ArrayWriter;
use super::tile_io::writing::TensorWriter;
use super::CubeMatmul;
use crate::matmul::tile_io::loading::tiled_layout::RowMajorTiling;
use crate::matmul::tile_io::loading::{LhsArrayLoader, RhsArrayLoader};
use crate::matmul::tile_io::writing::new_tensor_writer;

#[cube(launch_unchecked)]
pub(crate) fn matmul_instruction_launch<M: MatmulInstruction<I, O>, I: Numeric, O: Numeric>(
    lhs_array: Array<I>,
    rhs_array: Array<I>,
    mut out_array: Array<O>,
    #[comptime] layouts: (MatrixLayout, MatrixLayout),
) {
    let mut lhs = M::init_lhs(layouts.0);
    let mut rhs = M::init_rhs(layouts.1);
    let mut out = M::init_output();

    M::fill_lhs(lhs_array.as_slice(), &mut lhs);
    M::fill_rhs(rhs_array.as_slice(), &mut rhs);

    M::execute(&lhs, &rhs, &mut out);
    M::read_output(&out, out_array.as_slice_mut());
}

#[cube(launch_unchecked)]
pub(crate) fn block_matmul_launch<
    BM: BlockMatmul<
        Elem,
        LhsSmemTileReader<Elem, RowMajorTiling>,
        RhsSmemTileReader<Elem, RowMajorTiling>,
        ArrayWriter<Elem>,
    >,
    Elem: Numeric,
>(
    lhs_data: Array<Line<Elem>>,
    rhs_data: Array<Line<Elem>>,
    out_result: Array<Line<Elem>>,
    #[comptime] layouts: (MatrixLayout, MatrixLayout),
    #[comptime] block_info: BlockInfos,
) {
    let lhs_tile_reader = LhsArrayLoader::fill_block(
        &mut LhsArrayLoader::new(lhs_data, layouts.0.runtime(), block_info.lhs.runtime()),
        0,
    );

    let rhs_tile_reader = RhsArrayLoader::fill_block(
        &mut RhsArrayLoader::new(rhs_data, layouts.1.runtime(), block_info.rhs.runtime()),
        0,
    );

    let mut out_writer = ArrayWriter::<Elem> {
        gmem: out_result,
        block_info: block_info.out.runtime(),
    };

    let mut acc = BM::acc_init_zeros();
    BM::execute(&lhs_tile_reader, &rhs_tile_reader, &mut acc, layouts);
    BM::acc_read(&acc, &mut out_writer);
}

#[cube(launch_unchecked)]
pub(crate) fn cube_matmul_launch<
    CM: CubeMatmul<Elem, Lhs, Rhs, TensorWriter<Elem>>,
    Elem: Numeric,
    Lhs: Loader<Line<Elem>, Gmem = Tensor<Line<Elem>>>,
    Rhs: Loader<Line<Elem>, Gmem = Tensor<Line<Elem>>>,
>(
    lhs_tensor: Tensor<Line<Elem>>,
    rhs_tensor: Tensor<Line<Elem>>,
    out_tensor: Tensor<Line<Elem>>,
    #[comptime] layouts: (MatrixLayout, MatrixLayout),
    #[comptime] block_info: BlockInfos,
) {
    let k = lhs_tensor.shape(lhs_tensor.rank() - 1);

    let lhs_loader = Lhs::new(lhs_tensor, layouts.0, block_info.lhs);
    let rhs_loader = Rhs::new(rhs_tensor, layouts.1, block_info.rhs);
    let out = new_tensor_writer(out_tensor, block_info.out);

    CM::execute(lhs_loader, rhs_loader, out, (0, k), layouts);
}
