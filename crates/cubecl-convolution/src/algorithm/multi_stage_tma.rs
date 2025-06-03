use std::marker::PhantomData;

use cubecl_core::{
    CubeCount, CubeDim, Runtime,
    client::ComputeClient,
    ir::Elem,
    prelude::{Numeric, TensorHandleRef},
};

use crate::{
    base::{ConvolutionConfigFactory, ConvolutionProblem},
    homogeneous::multi_stage_tma::MultiStageTmaConvolutionFamily,
    selection::convolution_matmul_selection,
};

use cubecl_matmul::{
    components::{
        InputIdent, InvalidConfigError, MatmulLineSizes, MatmulPrecision,
        global::args::TensorMapArgs,
        stage::{FullReaderFamily, NumStages, plane_matmul::PlaneMatmulFamily},
        tile::TileMatmulFamily,
    },
    kernels::matmul::MatmulSelection,
};

use cubecl_std::tensor::TensorHandle;

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
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, FullReaderFamily, FullReaderFamily>;
    type GlobalConvolution = MultiStageTmaConvolutionFamily<Self::StageMatmul>;

    type Args = TensorMapArgs;

    fn cube_dim(selection: &MatmulSelection) -> CubeDim {
        CubeDim::new(
            selection.plane_dim,
            selection.tiling_scheme.tiles_in_stage_m(),
            1,
        )
    }

    fn cube_count(selection: &MatmulSelection, problem: &ConvolutionProblem) -> CubeCount {
        let m_stage = selection.tiling_scheme.elements_in_stage_m();
        let n_stage = selection.tiling_scheme.elements_in_stage_n();
        let cubes_needed_m = (problem.m as u32).div_ceil(m_stage);
        let cubes_needed_n = (problem.n as u32).div_ceil(n_stage);

        CubeCount::Static(cubes_needed_m, cubes_needed_n, 1)
    }

    fn make_config<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        input: <Self::GlobalConvolution as ConvolutionConfigFactory>::Input,
        problem: &ConvolutionProblem,
        line_sizes: &MatmulLineSizes,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
    ) -> Result<<Self::GlobalConvolution as ConvolutionConfigFactory>::Config, InvalidConfigError>
    {
        check_problem_tma(problem)?;

        let config = Self::GlobalConvolution::make_config::<R, MP>(
            client, input, problem, line_sizes, cube_dim, cube_count,
        );
        Self::GlobalConvolution::check_config(&config)?;
        Ok(config)
    }

    fn check_availability<R: Runtime, MP: cubecl_matmul::components::MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &<Self::GlobalConvolution as ConvolutionConfigFactory>::Config,
    ) -> Result<(), cubecl_matmul::kernels::MatmulAvailabilityError> {
        <Self::GlobalConvolution as ConvolutionConfigFactory>::check_availability::<R, MP>(
            client, config,
        )?;

        if !client
            .properties()
            .feature_enabled(cubecl_core::Feature::Tma(cubecl_core::TmaFeature::Base))
        {
            return Err(cubecl_matmul::kernels::MatmulAvailabilityError::TmaUnavailable);
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
    fn num_stages() -> NumStages {
        (1, 1).into()
    }

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &ConvolutionProblem,
        plane_dim: u32,
        elem_stage: Elem,
        elem_acc: Elem,
    ) -> MatmulSelection {
        convolution_matmul_selection::<TMM, R>(client, problem, plane_dim, elem_stage, elem_acc)
    }
}
