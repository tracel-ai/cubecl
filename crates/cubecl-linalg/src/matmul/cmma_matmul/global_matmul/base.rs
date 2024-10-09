use crate::matmul::data::TensorView;
use crate::matmul::launch::cube_matmul_launch;
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::problem::{MatmulProblem, Requirements};
use crate::matmul::stage_info::StageInfos;
use crate::matmul::tile_io::writing::TensorWriter;
use crate::matmul::tile_io::Loader;
use crate::matmul::{StageMatmul, GlobalMatmul, Matmul, TensorMatmul};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

pub struct CmmaGlobalMatmul<
    Elem: Numeric,
    BM: StageMatmul<Elem, Lhs::StageReader, Rhs::StageReader, TensorWriter<Elem>>,
    Lhs: Loader<Elem>,
    Rhs: Loader<Elem>,
> {
    _elem: PhantomData<Elem>,
    _block_matmul: PhantomData<BM>,
    _lhs: PhantomData<Lhs>,
    _rhs: PhantomData<Rhs>,
}

#[cube]
impl<
        Elem: Numeric,
        BM: StageMatmul<Elem, Lhs::StageReader, Rhs::StageReader, TensorWriter<Elem>>,
        Lhs: Loader<Elem, GlobalView = TensorView<Elem>>,
        Rhs: Loader<Elem, GlobalView = TensorView<Elem>>,
    > GlobalMatmul<Elem, Lhs, Rhs, TensorWriter<Elem>> for CmmaGlobalMatmul<Elem, BM, Lhs, Rhs>
{
    fn execute(
        mut lhs_loader: Lhs,
        mut rhs_loader: Rhs,
        mut out_writer: TensorWriter<Elem>,
        k_range: (u32, u32),
    ) {
        let k_step = BM::K;
        let range = k_range.1 - k_range.0;
        let num_loops = (range + k_step - 1) / k_step;

        let mut acc = BM::acc_init_zeros();
        Lhs::init_view(&mut lhs_loader, CUBE_POS_X, k_range.0);
        Rhs::init_view(&mut rhs_loader, CUBE_POS_Y, k_range.0);

        for _ in 0..num_loops {
            BM::execute(
                &Lhs::fill_block(&mut lhs_loader),
                &Rhs::fill_block(&mut rhs_loader),
                &mut acc,
            );

            Lhs::advance_view(&mut lhs_loader, k_step);
            Rhs::advance_view(&mut rhs_loader, k_step);
        }

        BM::acc_read(&acc, &mut out_writer);
    }
}

impl<
        Elem: Numeric,
        BM: StageMatmul<Elem, Lhs::StageReader, Rhs::StageReader, TensorWriter<Elem>>,
        Lhs: Loader<Elem>,
        Rhs: Loader<Elem>,
    > Matmul<Elem, Elem> for CmmaGlobalMatmul<Elem, BM, Lhs, Rhs>
{
    fn can_process(problem: MatmulProblem) -> bool {
        problem.m <= BM::M && problem.n <= BM::N
    }

    fn requirements(problem: MatmulProblem) -> Requirements {
        BM::requirements(problem)
    }

    fn stage_infos() -> StageInfos {
        BM::stage_infos()
    }
}

impl<
        Elem: Numeric,
        BM: StageMatmul<Elem, Lhs::StageReader, Rhs::StageReader, TensorWriter<Elem>>,
        Lhs: Loader<Elem, GlobalView = TensorView<Elem>>,
        Rhs: Loader<Elem, GlobalView = TensorView<Elem>>,
    > TensorMatmul<Elem> for CmmaGlobalMatmul<Elem, BM, Lhs, Rhs>
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
            Self::stage_infos(),
        );
    }
}
