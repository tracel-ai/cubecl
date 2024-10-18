use crate::matmul::cmma_matmul::stage::{LhsStageReader, OutStageWriter, RhsStageReader};
use crate::matmul::launch::cube_matmul_launch;
use crate::matmul::matmul_global::{GlobalMatmul, Loader};
use crate::matmul::matmul_global::{GmmConfig, Unloader};
use crate::matmul::matmul_stage::{SmmConfig, StageMatmul};
use crate::matmul::{Matmul, MatmulLaunch};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::{CmmaGlobalMatmulConfig, LhsTensorLoader, RhsTensorLoader, TensorUnloader};

pub struct CmmaGlobalMatmul<
    EG: Numeric,
    ES: Numeric,
    SMM: StageMatmul<
        ES,
        EG,
        LhsStageReader<ES, G::SmmConfig>,
        RhsStageReader<ES, G::SmmConfig>,
        OutStageWriter<EG>,
        G::SmmConfig,
    >,
    G: GmmConfig,
> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
    _stage_matmul: PhantomData<SMM>,
    _config: PhantomData<G>,
}

#[cube]
impl<EG, ES, SMM, G>
    GlobalMatmul<
        EG,
        ES,
        LhsTensorLoader<EG, ES, G>,
        RhsTensorLoader<EG, ES, G>,
        TensorUnloader<EG, G>,
        G,
    > for CmmaGlobalMatmul<EG, ES, SMM, G>
where
    EG: Numeric,
    ES: Numeric,
    SMM: StageMatmul<
        ES,
        EG,
        LhsStageReader<ES, G::SmmConfig>,
        RhsStageReader<ES, G::SmmConfig>,
        OutStageWriter<EG>,
        G::SmmConfig,
    >,
    G: GmmConfig,
{
    fn execute(
        mut lhs_loader: LhsTensorLoader<EG, ES, G>,
        mut rhs_loader: RhsTensorLoader<EG, ES, G>,
        out_unloader: TensorUnloader<EG, G>,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let k_step = SMM::K;
        let range = k_range.1 - k_range.0;
        let num_loops = (range + k_step - 1) / k_step;

        let mut acc = SMM::acc_init_zeros();

        // TODO cube mapper
        LhsTensorLoader::init_view(&mut lhs_loader, CUBE_POS_X * SMM::M, k_range.0);
        RhsTensorLoader::init_view(&mut rhs_loader, CUBE_POS_Y * SMM::N, k_range.0);

        // TODO init_view for Out or it will always start at (0,0)

        for _ in 0..num_loops {
            SMM::execute(
                &LhsTensorLoader::fill_stage(&mut lhs_loader, config),
                &RhsTensorLoader::fill_stage(&mut rhs_loader, config),
                &mut acc,
                config.to_smm_config(),
            );

            LhsTensorLoader::advance_view(&mut lhs_loader, k_step);
            RhsTensorLoader::advance_view(&mut rhs_loader, k_step);
        }

        SMM::acc_read::<G>(
            &acc,
            &mut TensorUnloader::unload(out_unloader),
            config.to_smm_config(),
            config,
        );
    }
}

impl<EG, ES, SMM, G> Matmul<EG, EG> for CmmaGlobalMatmul<EG, ES, SMM, G>
where
    EG: Numeric,
    ES: Numeric,
    SMM: StageMatmul<
        ES,
        EG,
        LhsStageReader<ES, G::SmmConfig>,
        RhsStageReader<ES, G::SmmConfig>,
        OutStageWriter<EG>,
        G::SmmConfig,
    >,
    G: GmmConfig,
{
    type Config = G;

    fn check_config(config: Self::Config) {
        SMM::check_config(config.to_smm_config());
    }
}

impl<EG, ES, SMM, S: SmmConfig> MatmulLaunch<EG, EG>
    for CmmaGlobalMatmul<EG, ES, SMM, CmmaGlobalMatmulConfig<S>>
where
    EG: Numeric,
    ES: Numeric,
    SMM: StageMatmul<ES, EG, LhsStageReader<ES, S>, RhsStageReader<ES, S>, OutStageWriter<EG>, S>,
{
    type MatmulLaunchConfig = CmmaGlobalMatmulConfig<S>;

    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        lhs: TensorArg<'_, R>,
        rhs: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        config: CmmaGlobalMatmulConfig<S>,
    ) {
        Self::check_config(config);
        cube_matmul_launch::launch_unchecked::<EG, ES, Self, CmmaGlobalMatmulConfig<S>, R>(
            &client, cube_count, cube_dim, lhs, rhs, out, config,
        );
    }
}
