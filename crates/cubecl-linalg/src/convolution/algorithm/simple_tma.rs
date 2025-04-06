use std::marker::PhantomData;

use cubecl_core::{CubeCount, CubeDim};

use crate::{
    convolution::{base::ConvolutionProblem, homogeneous::simple_tma::SimpleTmaConvolutionFamily},
    matmul::components::{MatmulSelection, stage, tile::TileMatmulFamily},
};

use super::Algorithm;

/// Cmma convolution
pub struct SimpleTmaConvAlgorithm<TMM: TileMatmulFamily> {
    _tmm: PhantomData<TMM>,
}

impl<TMM: TileMatmulFamily> Algorithm for SimpleTmaConvAlgorithm<TMM> {
    type TileMatmul = TMM;
    type StageMatmul = stage::multi_buffer::MultiBufferMatmulFamily<Self::TileMatmul>;
    type GlobalConvolution = SimpleTmaConvolutionFamily<Self::StageMatmul>;

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
}
