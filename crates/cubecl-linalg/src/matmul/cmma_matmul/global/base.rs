use crate::matmul::cmma_matmul::config::{CmmaConfig, CmmaPreConfig};
use crate::matmul::launch::cube_matmul_launch;
use crate::matmul::matmul_global::{GlobalMatmul, Loader, Unloader};
use crate::matmul::matmul_global::{GmmConfig, TensorView};
use crate::matmul::matmul_stage::StageMatmul;
use crate::matmul::Matmul;
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
        Lhs: Loader<EG, ES, ReadView = TensorView<EG>, Config = CmmaConfig>,
        Rhs: Loader<EG, ES, ReadView = TensorView<EG>, Config = CmmaConfig>,
        Out: Unloader<EG, WriteView = TensorView<EG>>,
    > GlobalMatmul<EG, ES, Lhs, Rhs, Out> for CmmaGlobalMatmul<EG, ES, SMM, Lhs, Rhs, Out>
{
    fn execute(
        mut lhs_loader: Lhs,
        mut rhs_loader: Rhs,
        out_unloader: Out,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
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
                &Lhs::fill_stage(&mut lhs_loader, config),
                &Rhs::fill_stage(&mut rhs_loader, config),
                &mut acc,
                config
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
        SMM: StageMatmul<
            ES,
            EG,
            Lhs::StageReader,
            Rhs::StageReader,
            Out::StageWriter,
            Config = CmmaConfig,
        >,
        Lhs: Loader<EG, ES, ReadView = TensorView<EG>, Config = CmmaConfig>,
        Rhs: Loader<EG, ES, ReadView = TensorView<EG>, Config = CmmaConfig>,
        Out: Unloader<EG, WriteView = TensorView<EG>>,
    > Matmul<EG, EG> for CmmaGlobalMatmul<EG, ES, SMM, Lhs, Rhs, Out>
{
    type Config = CmmaConfig;

    fn preconfigure() -> CmmaPreConfig {
        SMM::preconfigure()
    }

    fn check_config(config: Self::Config) {
        SMM::check_config(config.into_smm_config());
    }

    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        lhs: TensorArg<'_, R>,
        rhs: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        config: Self::Config,
    ) {
        Self::check_config(config);
        cube_matmul_launch::launch_unchecked::<EG, ES, Self, Lhs, Rhs, Out, R>(
            &client,
            cube_count,
            cube_dim,
            lhs,
            rhs,
            out,
            config.layouts,
            config,
        );
    }
}
