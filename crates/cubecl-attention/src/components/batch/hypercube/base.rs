use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::{AttentionBlueprint, AttentionProblem};

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct HypercubeBlueprint {}

impl HypercubeBlueprint {
    pub fn to_hypercube_config(&self) -> HypercubeConfig {
        HypercubeConfig {}
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct HypercubeConfig {}

impl HypercubeConfig {
    pub fn cube_count_plan(
        &self,
        problem: &AttentionProblem,
        blueprint: &AttentionBlueprint,
    ) -> CubeCountPlan {
        CubeCountPlan {
            inner: (problem.seq_q as u32).div_ceil(
                blueprint.tiling_scheme.tile_size.seq_q
                    * blueprint.tiling_scheme.partition_size.seq_q
                    * blueprint.tiling_scheme.stage_size.seq_q,
            ),
            outer: (problem.batch * problem.num_heads) as u32,
        }
    }
}

pub struct CubeCountPlan {
    inner: u32,
    outer: u32,
}

impl CubeCountPlan {
    pub fn resolve(&self) -> CubeCount {
        CubeCount::Static(self.inner, self.outer, 1)
    }

    /// Make a CubeCountInput from CubeCountPlan
    pub fn as_args<'a, R: Runtime>(&self) -> CubeCountInputArgs<'a, R> {
        CubeCountInputArgs::Tmp {
            dummy: ScalarArg::new(0),
        }
    }
}

#[derive(CubeType, CubeLaunch)]
/// CubeCountPlan stripped of non-essential runtime information
pub enum CubeCountInput {
    Tmp { dummy: u32 },
}
