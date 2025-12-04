use cubecl_core::CubeDim;

use crate::components::{
    AttentionCheckBounds, AttentionLineSizes, AttentionProblem, AttentionTilingScheme,
    batch::{CubeCountPlan, HypercubeBlueprint},
};

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct AttentionBlueprint {
    pub hypercube_blueprint: HypercubeBlueprint,

    pub tiling_scheme: AttentionTilingScheme,
    pub plane_dim: u32,
    pub num_planes: u32,

    pub reuse_key_value: bool,
    pub two_rows_in_array_tile: bool,

    pub line_sizes: AttentionLineSizes,

    pub masked: bool,
    pub causal: bool,

    pub check_bounds: AttentionCheckBounds,
}

impl AttentionBlueprint {
    pub fn cube_dim(&self) -> CubeDim {
        CubeDim {
            x: self.plane_dim,
            y: self.num_planes,
            z: 1,
        }
    }

    pub fn cube_count_plan(&self, problem: &AttentionProblem) -> CubeCountPlan {
        self.hypercube_blueprint
            .to_hypercube_config()
            .cube_count_plan(problem, self)
    }
}
