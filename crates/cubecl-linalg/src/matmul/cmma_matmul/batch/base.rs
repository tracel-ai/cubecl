use std::marker::PhantomData;

use crate::matmul::cmma_matmul::global::{LhsTensorLoader, RhsTensorLoader, TensorUnloader};
use crate::matmul::matmul_batch::{BatchMatmul, BmmConfig};
use crate::matmul::matmul_global::GlobalMatmul;
use crate::matmul::matrix::Ident;
use crate::matmul::{Matmul, MatmulLaunch};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

/// Performs matrix multiplication at the batch level,
/// with one cube assigned to each underlying global matmul
pub struct OneToOneBatchMatmul<
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
    > BatchMatmul<EG, B> for OneToOneBatchMatmul<EG, ES, GMM, B>
{
    fn execute(
        lhs: Tensor<Line<EG>>,
        rhs: Tensor<Line<EG>>,
        out: Tensor<Line<EG>>,
        #[comptime] config: Self::Config,
    ) {
        // TODO row/col/swizzle
        let x_offset = CUBE_POS_X * config.stage_dim(Ident::Lhs).num_elements_x_dim();
        let y_offset = CUBE_POS_Y * config.stage_dim(Ident::Rhs).num_elements_y_dim();
        let nth_batch = CUBE_POS_Z;
        let k_range = (0, lhs.shape(lhs.rank() - 1));

        GMM::execute(
            LhsTensorLoader::new(lhs, x_offset, k_range.0, nth_batch, config.to_gmm_config()),
            RhsTensorLoader::new(rhs, k_range.0, y_offset, nth_batch, config.to_gmm_config()),
            TensorUnloader::new(out, x_offset, y_offset, nth_batch),
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
    > Matmul<EG, EG> for OneToOneBatchMatmul<EG, ES, GMM, B>
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
    > MatmulLaunch<EG, EG> for OneToOneBatchMatmul<EG, ES, GMM, B>
{
    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        lhs: TensorArg<'_, R>,
        rhs: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        config: B,
    ) {
        Self::check_config(config);
        batch_matmul_launch::launch_unchecked::<EG, ES, Self, Self::Config, R>(
            &client, cube_count, cube_dim, lhs, rhs, out, config,
        );
    }
}

#[cube(launch_unchecked)]
pub(crate) fn batch_matmul_launch<
    EG: Numeric,
    ES: Numeric,
    BMM: BatchMatmul<EG, B>,
    B: BmmConfig,
>(
    lhs: Tensor<Line<EG>>,
    rhs: Tensor<Line<EG>>,
    out: Tensor<Line<EG>>,
    #[comptime] config: B,
) {
    BMM::execute(lhs, rhs, out, config);
}
