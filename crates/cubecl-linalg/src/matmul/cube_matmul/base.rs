use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_core::server::ComputeServer;

use crate::matmul::cmma_matmul::BlockInfos;
use crate::matmul::launch::cube_matmul_launch;
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::tile_io::loading::{
    LhsSmemTileReader, LhsTensorLoader, RhsSmemTileReader, RhsTensorLoader,
};
use crate::matmul::tile_io::writing::TensorWriter;
use crate::matmul::tile_io::Loader;
use crate::matmul::{BlockMatmul, CubeMatmul, Matmul, TensorMatmul};

pub struct CmmaCubeMatmul<
    Elem: Numeric,
    BM: BlockMatmul<Elem, LhsSmemTileReader<Elem>, RhsSmemTileReader<Elem>, TensorWriter<Elem>>,
> {
    _elem: PhantomData<Elem>,
    _block_matmul: PhantomData<BM>,
}

impl<
        Elem: Numeric,
        BM: BlockMatmul<Elem, LhsSmemTileReader<Elem>, RhsSmemTileReader<Elem>, TensorWriter<Elem>>,
    > Matmul<Elem, Elem> for CmmaCubeMatmul<Elem, BM>
{
    fn cube_dim_resources() -> CubeDim {
        BM::cube_dim_resources()
    }

    fn cube_count_resources<S: ComputeServer>() -> CubeCount<S> {
        CubeCount::Static(1, 1, 1)
    }

    fn block_infos() -> BlockInfos {
        BM::block_infos()
    }
}

impl<
        Elem: Numeric,
        BM: BlockMatmul<Elem, LhsSmemTileReader<Elem>, RhsSmemTileReader<Elem>, TensorWriter<Elem>>,
    > TensorMatmul<Elem> for CmmaCubeMatmul<Elem, BM>
{
    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount<<R as Runtime>::Server>,
        lhs: TensorArg<'_, R>,
        rhs: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        layouts: (MatrixLayout, MatrixLayout),
    ) {
        cube_matmul_launch::launch_unchecked::<Self, Elem, R>(
            &client,
            cube_count,
            cube_dim,
            lhs,
            rhs,
            out,
            layouts,
            Self::block_infos(),
        );
    }
}

#[cube]
impl<
        Elem: Numeric,
        BM: BlockMatmul<Elem, LhsSmemTileReader<Elem>, RhsSmemTileReader<Elem>, TensorWriter<Elem>>,
    > CubeMatmul<Elem, LhsTensorLoader<Elem>, RhsTensorLoader<Elem>, TensorWriter<Elem>>
    for CmmaCubeMatmul<Elem, BM>
{
    fn execute(
        mut lhs_reader: LhsTensorLoader<Elem>,
        mut rhs_reader: RhsTensorLoader<Elem>,
        mut out_writer: TensorWriter<Elem>,
        k_range: (u32, u32),
        layouts: (MatrixLayout, MatrixLayout),
    ) {
        let k_step = BM::K;
        let range = k_range.1 - k_range.0;
        let num_loops = (range + k_step - 1) / k_step;

        let mut acc = BM::acc_init_zeros();

        for block_iter in 0..num_loops {
            let k_offset = block_iter * k_step;

            BM::execute(
                &LhsTensorLoader::load_block(&mut lhs_reader, k_offset),
                &RhsTensorLoader::load_block(&mut rhs_reader, k_offset),
                &mut acc,
                layouts,
            );
        }

        BM::acc_read(&acc, &mut out_writer);
    }
}
