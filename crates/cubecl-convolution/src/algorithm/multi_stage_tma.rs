use std::marker::PhantomData;

use cubecl_core::{
    Runtime,
    client::ComputeClient,
    ir::Elem,
    prelude::{Numeric, TensorHandleRef},
};
use cubecl_runtime::storage::AllocError;

use crate::{
    base::ConvolutionProblem, homogeneous::multi_stage_tma::MultiStageTmaConvolutionFamily,
    selection::convolution_matmul_selection,
};

use cubecl_matmul::components::{
    InputIdent, MatmulSelection,
    global::args::TensorMapArgs,
    stage::{FullReaderFamily, NumStages, PlaneMatmulFamily},
    tile::TileMatmulFamily,
};

use cubecl_std::tensor::TensorHandle;

use super::{Algorithm, simple_tma::into_tensor_handle_tma};

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

    fn into_tensor_handle<R: Runtime, E: Numeric>(
        client: &ComputeClient<R::Server, R::Channel>,
        handle: &TensorHandleRef<'_, R>,
        ident: InputIdent,
    ) -> Result<TensorHandle<R, E>, AllocError> {
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
