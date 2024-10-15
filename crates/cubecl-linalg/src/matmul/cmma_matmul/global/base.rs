use crate::matmul::cmma_matmul::config::CmmaConfig;
use crate::matmul::launch::cube_matmul_launch;
use crate::matmul::matmul_global::TensorView;
use crate::matmul::matmul_global::{GlobalMatmul, Loader, Unloader};
use crate::matmul::matmul_stage::StageMatmul;
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::problem::{MatmulProblem, Requirements};
use crate::matmul::stage_info::StageInfos;
use crate::matmul::{Matmul, TensorMatmul};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

pub struct CmmaGlobalMatmul<
    EG: Numeric,
    ES: Numeric,
    BM: StageMatmul<ES, EG, Lhs::StageReader, Rhs::StageReader, Out::StageWriter>,
    Lhs: Loader<EG, ES>,
    Rhs: Loader<EG, ES>,
    Out: Unloader<EG>,
> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
    _block_matmul: PhantomData<BM>,
    _lhs: PhantomData<Lhs>,
    _rhs: PhantomData<Rhs>,
    _out: PhantomData<Out>,
}

#[cube]
impl<
        EG: Numeric,
        ES: Numeric,
        SMM: StageMatmul<
            ES,
            EG,
            Lhs::StageReader,
            Rhs::StageReader,
            Out::StageWriter,
            Config = CmmaConfig,
        >,
        Lhs: Loader<EG, ES, GlobalView = TensorView<EG>>,
        Rhs: Loader<EG, ES, GlobalView = TensorView<EG>>,
        Out: Unloader<EG, GlobalView = TensorView<EG>>,
    > GlobalMatmul<EG, ES, Lhs, Rhs, Out> for CmmaGlobalMatmul<EG, ES, SMM, Lhs, Rhs, Out>
{
    fn execute(
        mut lhs_loader: Lhs,
        mut rhs_loader: Rhs,
        out_unloader: Out,
        k_range: (u32, u32),
        #[comptime] config: &Self::Config,
    ) {
        let k_step = SMM::K;
        let range = k_range.1 - k_range.0;
        let num_loops = (range + k_step - 1) / k_step;

        let mut acc = SMM::acc_init_zeros();

        // TODO cube mapper
        Lhs::init_view(&mut lhs_loader, CUBE_POS_X * SMM::M, k_range.0);
        Rhs::init_view(&mut rhs_loader, CUBE_POS_Y * SMM::N, k_range.0);

        // TODO init_view for Out or it will always start at (0,0)

        for _ in 0..num_loops {
            SMM::execute(
                &Lhs::fill_block(&mut lhs_loader),
                &Rhs::fill_block(&mut rhs_loader),
                &mut acc,
            );

            Lhs::advance_view(&mut lhs_loader, k_step);
            Rhs::advance_view(&mut rhs_loader, k_step);
        }

        SMM::acc_read(&acc, &mut Out::unload(out_unloader), config);
    }
}

impl<
        EG: Numeric,
        ES: Numeric,
        BM: StageMatmul<ES, EG, Lhs::StageReader, Rhs::StageReader, Out::StageWriter>,
        Lhs: Loader<EG, ES>,
        Rhs: Loader<EG, ES>,
        Out: Unloader<EG>,
    > Matmul<EG, EG> for CmmaGlobalMatmul<EG, ES, BM, Lhs, Rhs, Out>
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
        EG: Numeric,
        ES: Numeric,
        SMM: StageMatmul<
            ES,
            EG,
            Lhs::StageReader,
            Rhs::StageReader,
            Out::StageWriter,
            Config = CmmaConfig,
        >,
        Lhs: Loader<EG, ES, GlobalView = TensorView<EG>>,
        Rhs: Loader<EG, ES, GlobalView = TensorView<EG>>,
        Out: Unloader<EG, GlobalView = TensorView<EG>>,
    > TensorMatmul<EG> for CmmaGlobalMatmul<EG, ES, SMM, Lhs, Rhs, Out>
{
    type Config = CmmaConfig;

    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        lhs: TensorArg<'_, R>,
        rhs: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        layouts: (MatrixLayout, MatrixLayout),
        config: CmmaConfig,
    ) {
        cube_matmul_launch::launch_unchecked::<EG, ES, Self, Lhs, Rhs, Out, R>(
            &client,
            cube_count,
            cube_dim,
            lhs,
            rhs,
            out,
            layouts,
            Self::stage_infos(),
            config,
        );
    }
}
