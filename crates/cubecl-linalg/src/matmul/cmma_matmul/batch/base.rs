use std::marker::PhantomData;

use crate::matmul::cmma_matmul::global::{
    new_lhs_tensor_loader, new_rhs_tensor_loader, new_tensor_unloader, LhsTensorLoader,
    RhsTensorLoader, TensorUnloader,
};
use crate::matmul::launch::batch_matmul_launch;
use crate::matmul::matmul_batch::{BatchMatmul, BmmConfig};
use crate::matmul::matmul_global::GlobalMatmul;
use crate::matmul::matrix::Ident;
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
        // TODO this is naive
        let x_offset = CUBE_POS_X * config.stage_dim(Ident::Lhs).num_elements_x_dim();
        let y_offset = CUBE_POS_Y * config.stage_dim(Ident::Rhs).num_elements_y_dim();
        let k_range = (0, lhs.shape(lhs.rank() - 1));

        GMM::execute(
            new_lhs_tensor_loader(lhs, x_offset, k_range.0, config.to_gmm_config()),
            new_rhs_tensor_loader(rhs, k_range.0, y_offset, config.to_gmm_config()),
            new_tensor_unloader(out, x_offset, y_offset),
            x_offset,
            y_offset,
            k_range,
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
