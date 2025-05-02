use std::marker::PhantomData;

use cubecl_core::{
    CubeCount, CubeDim, Runtime,
    client::ComputeClient,
    prelude::{Numeric, TensorHandleRef},
};

use crate::{
    convolution::{base::ConvolutionProblem, homogeneous::simple::SimpleConvolutionFamily},
    matmul::components::{
        InputIdent, MatmulSelection,
        global::args::TensorArgs,
        stage::{FullReaderFamily, plane_matmul::PlaneMatmulFamily},
        tile::TileMatmulFamily,
    },
    tensor::{TensorHandle, into_contiguous},
};

use super::Algorithm;

/// Cmma convolution
pub struct SimpleConvAlgorithm<TMM: TileMatmulFamily> {
    _tmm: PhantomData<TMM>,
}

impl<TMM: TileMatmulFamily> Algorithm for SimpleConvAlgorithm<TMM> {
    type TileMatmul = TMM;
    type StageMatmul = PlaneMatmulFamily<Self::TileMatmul, FullReaderFamily>;
    type GlobalConvolution = SimpleConvolutionFamily<Self::StageMatmul>;

    type Args = TensorArgs;

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

    fn into_tensor_handle<R: Runtime, E: Numeric>(
        client: &ComputeClient<R::Server, R::Channel>,
        handle: &TensorHandleRef<'_, R>,
        ident: crate::matmul::components::InputIdent,
    ) -> TensorHandle<R, E> {
        let rank = handle.shape.len();
        let dim_c = rank - 1;
        let mut handle = if has_valid_layout(handle, ident) {
            TensorHandle::from_ref(handle)
        } else {
            into_contiguous(client, handle)
        };
        match ident {
            InputIdent::Lhs => handle,
            InputIdent::Rhs => {
                // Reshape to (K, N) so the loader knows how to handle it
                handle.shape = vec![handle.shape[1..].iter().product(), handle.shape[0]];
                handle.strides = vec![handle.strides[dim_c], handle.strides[0]];
                handle
            }
        }
    }

    // TODO this is not the same as tma stages, it's stages in the sense of double buffering in matmul
    fn num_stages() -> u32 {
        1
    }
}

fn has_valid_layout<R: Runtime>(handle: &TensorHandleRef<'_, R>, ident: InputIdent) -> bool {
    let rank = handle.shape.len();
    let dim_c = rank - 1;
    match ident {
        InputIdent::Lhs => handle.strides[dim_c] == 1,
        InputIdent::Rhs => {
            let mut strides = handle.strides.to_vec();
            strides.sort();
            let ordered = handle.strides == strides;
            let mut contiguous_k = true;
            for i in 1..dim_c {
                contiguous_k &= strides[i] == strides[i + 1] * handle.shape[i + 1];
            }
            ordered && contiguous_k
        }
    }
}
