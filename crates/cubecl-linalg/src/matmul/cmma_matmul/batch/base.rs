use std::marker::PhantomData;

use crate::matmul::cmma_matmul::global::{
    new_lhs_tensor_loader, new_rhs_tensor_loader, new_tensor_unloader, LhsTensorLoader,
    RhsTensorLoader, TensorUnloader,
};
use crate::matmul::launch::batch_matmul_launch;
use crate::matmul::matmul_batch::{BatchMatmul, BmmConfig};
use crate::matmul::matmul_global::GlobalMatmul;
use crate::matmul::{Matmul, MatmulLaunch};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

pub struct CmmaBatchMatmul<
    EG: Numeric,
    ES: Numeric,
    GMM: GlobalMatmul<
        EG,
        ES,
        LhsTensorLoader<EG, ES, B::GmmConfig>,
        RhsTensorLoader<EG, ES, B::GmmConfig>,
        TensorUnloader<EG, B::GmmConfig>,
        B::GmmConfig,
    >,
    B: BmmConfig,
> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
    _gmm: PhantomData<GMM>,
    _config: PhantomData<B>,
}

#[cube]
impl<
        EG: Numeric,
        ES: Numeric,
        GMM: GlobalMatmul<
            EG,
            ES,
            LhsTensorLoader<EG, ES, B::GmmConfig>,
            RhsTensorLoader<EG, ES, B::GmmConfig>,
            TensorUnloader<EG, B::GmmConfig>,
            B::GmmConfig,
        >,
        B: BmmConfig,
    > BatchMatmul<EG, B> for CmmaBatchMatmul<EG, ES, GMM, B>
{
    fn execute(
        lhs: Tensor<Line<EG>>,
        rhs: Tensor<Line<EG>>,
        out: Tensor<Line<EG>>,
        #[comptime] config: Self::Config,
    ) {
        let k = lhs.shape(lhs.rank() - 1);

        let lhs_loader = new_lhs_tensor_loader(lhs, config.to_gmm_config());
        let rhs_loader = new_rhs_tensor_loader(rhs, config.to_gmm_config());
        let out_unloader = new_tensor_unloader(out);

        GMM::execute(
            lhs_loader,
            rhs_loader,
            out_unloader,
            (0, k),
            config.to_gmm_config(),
        );
    }
}

impl<
        EG: Numeric,
        ES: Numeric,
        GMM: GlobalMatmul<
            EG,
            ES,
            LhsTensorLoader<EG, ES, B::GmmConfig>,
            RhsTensorLoader<EG, ES, B::GmmConfig>,
            TensorUnloader<EG, B::GmmConfig>,
            B::GmmConfig,
        >,
        B: BmmConfig,
    > Matmul<EG, EG> for CmmaBatchMatmul<EG, ES, GMM, B>
{
    type Config = B;

    fn check_config(config: Self::Config) {
        GMM::check_config(config.to_gmm_config())
    }
}

impl<
        EG: Numeric,
        ES: Numeric,
        GMM: GlobalMatmul<
            EG,
            ES,
            LhsTensorLoader<EG, ES, B::GmmConfig>,
            RhsTensorLoader<EG, ES, B::GmmConfig>,
            TensorUnloader<EG, B::GmmConfig>,
            B::GmmConfig,
        >,
        B: BmmConfig,
    > MatmulLaunch<EG, EG> for CmmaBatchMatmul<EG, ES, GMM, B>
{
    type MatmulLaunchConfig = B;

    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        lhs: TensorArg<'_, R>,
        rhs: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        config: Self::MatmulLaunchConfig,
    ) {
        Self::check_config(config);
        batch_matmul_launch::launch_unchecked::<EG, ES, Self, Self::MatmulLaunchConfig, R>(
            &client, cube_count, cube_dim, lhs, rhs, out, config,
        );
    }
}
