use std::marker::PhantomData;

use cubecl_core::{
    Runtime,
    client::ComputeClient,
    prelude::{Numeric, TensorHandleRef},
};
use cubecl_matmul::components::{MatmulElems, MatmulSelection, MatmulSetupError};

use crate::{
    base::ConvolutionProblem, homogeneous::simple::SimpleConvolutionFamily,
    selection::convolution_matmul_selection,
};
use cubecl_matmul::components::stage::NumStages;
use cubecl_matmul::components::{
    MatmulIdent,
    global::args::TensorArgs,
    stage::{FullReaderFamily, PlaneMatmulFamily},
    tile::TileMatmulFamily,
};

use cubecl_std::tensor::{TensorHandle, into_contiguous};

use super::Algorithm;

/// Cmma convolution
pub struct SimpleConvAlgorithm<TMM: TileMatmulFamily> {
    _tmm: PhantomData<TMM>,
}

impl<TMM: TileMatmulFamily> Algorithm for SimpleConvAlgorithm<TMM> {
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, FullReaderFamily, FullReaderFamily>;
    type GlobalConvolution = SimpleConvolutionFamily<Self::StageMatmul>;

    type Args = TensorArgs;

    fn into_tensor_handle<R: Runtime, E: Numeric>(
        client: &ComputeClient<R::Server, R::Channel>,
        handle: &TensorHandleRef<'_, R>,
        ident: MatmulIdent,
    ) -> TensorHandle<R, E> {
        let rank = handle.shape.len();
        let dim_c = rank - 1;
        let mut handle = if has_valid_layout(handle, ident) {
            TensorHandle::from_ref(handle)
        } else {
            into_contiguous(client, handle)
        };
        match ident {
            MatmulIdent::Lhs => handle,
            MatmulIdent::Rhs => {
                // Reshape to (K, N) so the loader knows how to handle it
                handle.shape = vec![handle.shape[1..].iter().product(), handle.shape[0]];
                handle.strides = vec![handle.strides[dim_c], handle.strides[0]];
                handle
            }
            MatmulIdent::Out => unreachable!(),
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
        MatmulIdent::Rhs => {
            let mut strides = handle.strides.to_vec();
            strides.sort();
            let ordered = handle.strides == strides;
            let mut contiguous_k = true;
            for i in 1..dim_c {
                contiguous_k &= strides[i] == strides[i + 1] * handle.shape[i + 1];
            }
            ordered && contiguous_k
        }
        MatmulIdent::Out => unreachable!(),
    }
}
