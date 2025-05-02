use std::marker::PhantomData;

use cubecl_core::{
    CubeCount, CubeDim, Runtime,
    client::ComputeClient,
    prelude::{Numeric, TensorHandleRef},
};

use crate::{
    convolution::{
        base::{ConvolutionConfigFactory, ConvolutionProblem},
        homogeneous::multi_stage_tma::MultiStageTmaConvolutionFamily,
    },
    matmul::components::{
        InputIdent, InvalidConfigError, MatmulPrecision, MatmulSelection,
        global::args::TensorMapArgs,
        stage::{FullReaderFamily, plane_matmul::PlaneMatmulFamily},
        tile::TileMatmulFamily,
    },
    tensor::TensorHandle,
};

use super::{
    Algorithm,
    simple_tma::{check_problem_tma, into_tensor_handle_tma},
};

pub const TMA_STRIDE_ALIGN: usize = 16;

/// Cmma convolution
pub struct MultiStageTmaConvAlgorithm<TMM: TileMatmulFamily> {
    _tmm: PhantomData<TMM>,
}

impl<TMM: TileMatmulFamily> Algorithm for MultiStageTmaConvAlgorithm<TMM> {
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, FullReaderFamily>;
    type GlobalConvolution = MultiStageTmaConvolutionFamily<Self::StageMatmul>;

    type Args = TensorMapArgs;

    fn cube_dim(selection: &MatmulSelection) -> CubeDim {
        CubeDim::new(selection.plane_dim, selection.tile_count.m, 1)
    }

    fn cube_count(selection: &MatmulSelection, problem: &ConvolutionProblem) -> CubeCount {
        let m_stage = selection.tile_count.m * selection.tile_shape.m;
        let n_stage = selection.tile_count.n * selection.tile_shape.n;
        let cubes_needed_m = (problem.m as u32).div_ceil(m_stage);
        let cubes_needed_n = (problem.n as u32).div_ceil(n_stage);

        CubeCount::Static(cubes_needed_m, cubes_needed_n, 1)
    }

    fn make_config<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        input: <Self::GlobalConvolution as ConvolutionConfigFactory>::Input,
        problem: &ConvolutionProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
    ) -> Result<<Self::GlobalConvolution as ConvolutionConfigFactory>::Config, InvalidConfigError>
    {
        check_problem_tma(problem)?;

        let config = Self::GlobalConvolution::make_config::<R, MP>(
            client, input, problem, cube_dim, cube_count,
        );
        Self::GlobalConvolution::check_config(&config)?;
        Ok(config)
    }

    fn check_availability<R: Runtime, MP: crate::matmul::components::MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &<Self::GlobalConvolution as ConvolutionConfigFactory>::Config,
    ) -> Result<(), crate::matmul::kernels::MatmulAvailabilityError> {
        <Self::GlobalConvolution as ConvolutionConfigFactory>::check_availability::<R, MP>(
            client, config,
        )?;

        if !client
            .properties()
            .feature_enabled(cubecl_core::Feature::Tma(cubecl_core::TmaFeature::Base))
        {
            return Err(crate::matmul::kernels::MatmulAvailabilityError::TmaUnavailable);
        }

        Ok(())
    }

    fn into_tensor_handle<R: Runtime, E: Numeric>(
        client: &ComputeClient<R::Server, R::Channel>,
        handle: &TensorHandleRef<'_, R>,
        ident: InputIdent,
    ) -> TensorHandle<R, E> {
        into_tensor_handle_tma(client, handle, ident)
    }

    // TODO this is not the same as tma stages, it's stages in the sense of double buffering in matmul
    fn num_stages() -> u32 {
        1
    }
}
