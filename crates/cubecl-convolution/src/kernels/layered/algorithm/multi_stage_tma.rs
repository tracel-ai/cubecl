use std::marker::PhantomData;

use cubecl_core::{Runtime, client::ComputeClient, ir::StorageType, prelude::TensorHandleRef};

use cubecl_matmul::components::{
    MatmulElems, MatmulIdent, MatmulSelection, MatmulSetupError,
    global::args::TensorMapArgs,
    stage::{NumStages, PlaneMatmulFamily, StridedStageFamily},
    tile::{TileMatmulFamily, io::Strided},
};

use cubecl_std::{CubeOption, tensor::TensorHandle};

use crate::components::{
    ConvolutionProblem, convolution_matmul_selection,
    global::multi_stage::tma::MultiStageTmaConvolutionFamily,
};

use super::{Algorithm, simple_tma::into_tensor_handle_tma};

pub const TMA_STRIDE_ALIGN: usize = 16;

/// Cmma convolution
pub struct MultiStageTmaConvAlgorithm<TMM: TileMatmulFamily> {
    _tmm: PhantomData<TMM>,
}

impl<
    TMM: TileMatmulFamily<
            LhsTile = Strided,
            RhsTile = Strided,
            AccTile = CubeOption<Strided>,
            OutTile = Strided,
        >,
> Algorithm for MultiStageTmaConvAlgorithm<TMM>
{
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        StridedStageFamily,
        StridedStageFamily,
        Option<StridedStageFamily>,
    >;
    type GlobalConvolution = MultiStageTmaConvolutionFamily<Self::StageMatmul>;

    type Args = TensorMapArgs;

    fn into_tensor_handle<R: Runtime>(
        client: &ComputeClient<R::Server>,
        handle: &TensorHandleRef<'_, R>,
        ident: MatmulIdent,
        dtype: StorageType,
    ) -> TensorHandle<R> {
        into_tensor_handle_tma(client, handle, ident, dtype)
    }

    // TODO this is not the same as tma stages, it's stages in the sense of double buffering in matmul
    fn num_stages() -> NumStages {
        (1, 1).into()
    }

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &ConvolutionProblem,
        plane_dim: u32,
        matmul_elems: &mut MatmulElems,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        Ok(convolution_matmul_selection::<TMM, R>(
            client,
            problem,
            plane_dim,
            matmul_elems,
        ))
    }
}
