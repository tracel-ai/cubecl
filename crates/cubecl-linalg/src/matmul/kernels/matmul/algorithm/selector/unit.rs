use cubecl_core::{Runtime, client::ComputeClient, ir::Elem};

use crate::matmul::components::{MatmulProblem, MatmulSize, tile::TileMatmulFamily};

use super::MatmulSelection;

#[derive(Debug)]
pub struct UnitMatmulSelection {
    pub tile_shape: MatmulSize,
    pub tile_count: MatmulSize,
    pub plane_dim: u32,
}

impl MatmulSelection for UnitMatmulSelection {
    fn tile_shape(&self) -> MatmulSize {
        self.tile_shape
    }

    fn tile_count(&self) -> MatmulSize {
        self.tile_count
    }
}

pub fn unit_matmul_selection<TMM: TileMatmulFamily, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    problem: &MatmulProblem,
    plane_dim: u32,
    elem_stage: Elem,
    elem_acc: Elem,
) -> UnitMatmulSelection {
    // TODO
    UnitMatmulSelection {
        tile_shape: MatmulSize { m: 1, n: 1, k: 1 },
        tile_count: MatmulSize { m: 1, n: 1, k: 1 },
        plane_dim,
    }
}
