use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::block_info::BlockInfos;
use crate::matmul::data::TensorView;
use crate::matmul::launch::cube_matmul_launch;
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::problem::{MatmulProblem, Requirements};
use crate::matmul::tile_io::writing::TensorWriter;
use crate::matmul::tile_io::Loader;
use crate::matmul::{BlockMatmul, CubeMatmul, Matmul, TensorMatmul};

pub struct CmmaCubeMatmul<
    Elem: Numeric,
    BM: BlockMatmul<Elem, Lhs::BlockReader, Rhs::BlockReader, TensorWriter<Elem>>,
    Lhs: Loader<Elem>,
    Rhs: Loader<Elem>,
> {
    _elem: PhantomData<Elem>,
    _block_matmul: PhantomData<BM>,
    _lhs: PhantomData<Lhs>,
    _rhs: PhantomData<Rhs>,
}

impl<
        Elem: Numeric,
        BM: BlockMatmul<Elem, Lhs::BlockReader, Rhs::BlockReader, TensorWriter<Elem>>,
        Lhs: Loader<Elem>,
        Rhs: Loader<Elem>,
    > Matmul<Elem, Elem> for CmmaCubeMatmul<Elem, BM, Lhs, Rhs>
{
    fn can_process(problem: MatmulProblem) -> bool {
        problem.m <= BM::M && problem.n <= BM::N
    }

    fn requirements(problem: MatmulProblem) -> Requirements {
        BM::requirements(problem)
    }

    fn block_infos() -> BlockInfos {
        BM::block_infos()
    }
}

impl<
        Elem: Numeric,
        BM: BlockMatmul<Elem, Lhs::BlockReader, Rhs::BlockReader, TensorWriter<Elem>>,
        Lhs: Loader<Elem, GmemView = TensorView<Elem>>,
        Rhs: Loader<Elem, GmemView = TensorView<Elem>>,
    > TensorMatmul<Elem> for CmmaCubeMatmul<Elem, BM, Lhs, Rhs>
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
        cube_matmul_launch::launch_unchecked::<Self, Elem, Lhs, Rhs, R>(
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
        BM: BlockMatmul<Elem, Lhs::BlockReader, Rhs::BlockReader, TensorWriter<Elem>>,
        Lhs: Loader<Elem, GmemView = TensorView<Elem>>,
        Rhs: Loader<Elem, GmemView = TensorView<Elem>>,
    > CubeMatmul<Elem, Lhs, Rhs, TensorWriter<Elem>> for CmmaCubeMatmul<Elem, BM, Lhs, Rhs>
{
    fn execute(
        mut lhs_reader: Lhs,
        mut rhs_reader: Rhs,
        mut out_writer: TensorWriter<Elem>,
        k_range: (u32, u32),
    ) {
        let k_step = BM::K;
        let range = k_range.1 - k_range.0;
        let num_loops = (range + k_step - 1) / k_step;

        let mut acc = BM::acc_init_zeros();

        for block_iter in 0..num_loops {
            let k_offset = block_iter * k_step;

            BM::execute(
                &Lhs::fill_block(&mut lhs_reader),
                &Rhs::fill_block(&mut rhs_reader),
                &mut acc,
            );
        }

        BM::acc_read(&acc, &mut out_writer);
    }
}
