use crate::matmul::cmma_matmul::stage::{LhsStageReader, RhsStageReader};
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
        G::SmmConfig,
    >,
    G: GmmConfig,
{
    fn execute(
        mut lhs_loader: LhsTensorLoader<EG, ES, G>,
        mut rhs_loader: RhsTensorLoader<EG, ES, G>,
        mut out_unloader: TensorUnloader<EG, G>,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let k_step = SMM::K;
        let range = k_range.1 - k_range.0;
        let num_loops = (range + k_step - 1) / k_step;

        let mut acc = SMM::acc_init_zeros();

        // TODO cube mapper
        let x_offset = CUBE_POS_X * SMM::M;
        let y_offset = CUBE_POS_Y * SMM::N;

        // It could be tensor view directly
        LhsTensorLoader::init_view(&mut lhs_loader, x_offset, k_range.0);
        RhsTensorLoader::init_view(&mut rhs_loader, k_range.0, y_offset);
        TensorUnloader::init_view(&mut out_unloader, x_offset, y_offset);

        for _ in 0..num_loops {
            SMM::execute(
                &LhsTensorLoader::fill_stage(&mut lhs_loader, config),
                &RhsTensorLoader::fill_stage(&mut rhs_loader, config),
                &mut acc,
                config.to_smm_config(),
            );

            // It could be tensor view directly
            LhsTensorLoader::advance_view(&mut lhs_loader, k_step);
            RhsTensorLoader::advance_view(&mut rhs_loader, k_step);
        }

        // TODO TensorUnloader is bad abstraction. Doesn't do anything
        // The only thing loaders do more than view is
        // give stage readers and writers
        // OutStageWriter is bad, it plays on Gmm level in spite of its name
        // All it does is pass stuff around with a multiplication that somebody else could do

        // There should be only the TensorUnloader, which should replace the Out TensorView
        // And (L/R)hsTensorLoader should be merged with Input TensorView

        // BUT there should be a stage writer
        // It writes to the out smem
        // Then the unloader takes from the out smem to the global memory

        SMM::acc_read::<TensorUnloader<EG, G>, G>(
            &acc,
            &mut out_unloader,
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
    SMM: StageMatmul<ES, EG, LhsStageReader<ES, S>, RhsStageReader<ES, S>, S>,
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
