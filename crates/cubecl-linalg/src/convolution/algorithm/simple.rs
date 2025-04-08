use std::marker::PhantomData;

use cubecl_core::{CubeCount, CubeDim, Runtime, prelude::TensorHandleRef};

use crate::{
    convolution::{base::ConvolutionProblem, homogeneous::simple::SimpleConvolutionFamily},
    matmul::components::{
        MatmulSelection, global::args::TensorArgs, stage, tile::TileMatmulFamily,
    },
};

use super::Algorithm;

/// Cmma convolution
pub struct SimpleConvAlgorithm<TMM: TileMatmulFamily> {
    _tmm: PhantomData<TMM>,
}

impl<TMM: TileMatmulFamily> Algorithm for SimpleConvAlgorithm<TMM> {
    type TileMatmul = TMM;
    type StageMatmul = stage::multi_buffer::MultiBufferMatmulFamily<Self::TileMatmul>;
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

    fn has_valid_layout<R: Runtime>(handle: &TensorHandleRef<'_, R>) -> bool {
        let mut strides = handle.strides.to_vec();
        strides.sort();

        // Permuted strides
        if handle.strides != strides {
            return false;
        }

        // channels doesn't need to be contiguous with the rest of the tensor
        strides[2] * handle.shape[2] == strides[1]
            && strides[1] * handle.shape[1] == handle.strides[0]
    }
}
