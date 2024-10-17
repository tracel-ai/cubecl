use crate::matmul::cmma_matmul::stage::{LhsStageReader, OutStageWriter, RhsStageReader};
use crate::matmul::config::{ComptimeConfig, MatmulConfig};
use crate::matmul::launch::cube_matmul_launch;
use crate::matmul::matmul_global::{GlobalMatmul, Loader};
use crate::matmul::matmul_global::{GmmConfig, Unloader};
use crate::matmul::matmul_stage::{SmmConfig, StageMatmul};
use crate::matmul::Matmul;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::{LhsTensorLoader, RhsTensorLoader, TensorUnloader};

pub struct CmmaGlobalMatmul<
    EG: Numeric,
    ES: Numeric,
    SMM: StageMatmul<ES, EG, LhsStageReader<ES>, RhsStageReader<ES>, OutStageWriter<EG>>,
> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
    _stage_matmul: PhantomData<SMM>,
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct CmmaGlobalMatmulConfig<S: SmmConfig> {
    smm_config: S,
}

impl<S: SmmConfig> ComptimeConfig for CmmaGlobalMatmulConfig<S> {}

impl<S: SmmConfig> GmmConfig for CmmaGlobalMatmulConfig<S> {
    type SmmConfig = S;

    fn to_smm_config(&self) -> Self::SmmConfig {
        self.smm_config
    }
}

impl<S: SmmConfig> MatmulConfig for CmmaGlobalMatmulConfig<S> {
    fn num_planes(&self) -> u32 {
        todo!()
    }

    fn plane_dim(&self) -> u32 {
        todo!()
    }

    fn default(
        cube_dim: CubeDim,
        cube_count: CubeCount,
        problem: crate::matmul::problem::MatmulProblem,
    ) {
        todo!()
    }
}

#[cube]
impl<EG, ES, SMM>
    GlobalMatmul<EG, ES, LhsTensorLoader<EG, ES>, RhsTensorLoader<EG, ES>, TensorUnloader<EG>>
    for CmmaGlobalMatmul<EG, ES, SMM>
where
    EG: Numeric,
    ES: Numeric,
    SMM: StageMatmul<ES, EG, LhsStageReader<ES>, RhsStageReader<ES>, OutStageWriter<EG>>,
{
    fn execute(
        mut lhs_loader: LhsTensorLoader<EG, ES>,
        mut rhs_loader: RhsTensorLoader<EG, ES>,
        out_unloader: TensorUnloader<EG>,
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
                &LhsTensorLoader::fill_stage(&mut lhs_loader, config.to_smm_config()),
                &RhsTensorLoader::fill_stage(&mut rhs_loader, config.to_smm_config()),
                &mut acc,
                config.to_smm_config(),
            );

            LhsTensorLoader::advance_view(&mut lhs_loader, k_step);
            RhsTensorLoader::advance_view(&mut rhs_loader, k_step);
        }

        SMM::acc_read(
            &acc,
            &mut TensorUnloader::unload(out_unloader),
            config.to_smm_config(),
        );
    }
}

impl<EG, ES, SMM> Matmul<EG, EG> for CmmaGlobalMatmul<EG, ES, SMM>
where
    EG: Numeric,
    ES: Numeric,
    SMM: StageMatmul<ES, EG, LhsStageReader<ES>, RhsStageReader<ES>, OutStageWriter<EG>>,
{
    type Config = CmmaGlobalMatmulConfig<SMM::Config>;

    fn check_config(config: Self::Config) {
        SMM::check_config(config.to_smm_config());
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
        cube_matmul_launch::launch_unchecked::<EG, ES, Self, R>(
            &client, cube_count, cube_dim, lhs, rhs, out, config,
        );
    }
}
