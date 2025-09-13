use std::marker::PhantomData;

use cubecl_core::{
    Runtime,
    client::ComputeClient,
    prelude::{Numeric, TensorHandleRef},
};
use cubecl_matmul::components::{
    MatmulElems, MatmulSelection, MatmulSetupError, tile::loader::Strided,
};

use cubecl_matmul::components::stage::NumStages;
use cubecl_matmul::components::{
    MatmulIdent,
    global::args::TensorArgs,
    stage::{FullReaderFamily, PlaneMatmulFamily},
    tile::TileMatmulFamily,
};

use cubecl_std::{
    CubeOption,
    tensor::{TensorHandle, into_contiguous},
};

use crate::components::{
    ConvolutionProblem, convolution_matmul_selection,
    global::single_stage::simple::SimpleConvolutionFamily,
};

use super::Algorithm;

/// Cmma convolution
pub struct SimpleConvAlgorithm<TMM: TileMatmulFamily> {
    _tmm: PhantomData<TMM>,
}

impl<TMM: TileMatmulFamily<LhsTile = Strided, RhsTile = Strided, AccTile = CubeOption<Strided>>>
    Algorithm for SimpleConvAlgorithm<TMM>
{
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<
        Self::TileMatmul,
        FullReaderFamily,
        FullReaderFamily,
        Option<FullReaderFamily>,
    >;
    type GlobalConvolution = SimpleConvolutionFamily<Self::StageMatmul>;

    type Args = TensorArgs;

    fn into_tensor_handle<R: Runtime, E: Numeric>(
        client: &ComputeClient<R::Server, R::Channel>,
        handle: &TensorHandleRef<'_, R>,
        ident: MatmulIdent,
    ) -> TensorHandle<R, E> {
        if has_valid_layout(handle, ident) {
            TensorHandle::from_ref(handle)
        } else {
            into_contiguous(client, handle)
        }
    }

    // TODO this is not the same as tma stages, it's stages in the sense of double buffering in matmul
    fn num_stages() -> NumStages {
        (1, 1).into()
    }

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &ConvolutionProblem,
        plane_dim: u32,
        matmul_elems: MatmulElems,
    ) -> Result<MatmulSelection, MatmulSetupError> {
        Ok(convolution_matmul_selection::<TMM, R>(
            client,
            problem,
            plane_dim,
            matmul_elems,
        ))
    }
}

fn has_valid_layout<R: Runtime>(handle: &TensorHandleRef<'_, R>, ident: MatmulIdent) -> bool {
    let rank = handle.shape.len();
    let dim_c = rank - 1;
    match ident {
        MatmulIdent::Lhs => handle.strides[dim_c] == 1,
        MatmulIdent::Rhs => handle.strides[dim_c] == 1,
        MatmulIdent::Out => unreachable!(),
    }
}
