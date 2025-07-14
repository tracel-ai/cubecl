use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::AttentionProblem;

pub struct HypercubeConfig {}

impl HypercubeConfig {
    pub fn cube_count_plan(
        &self,
        _problem: &AttentionProblem,
        _max_cube_count: CubeCount,
    ) -> CubeCountPlan {
        CubeCountPlan {}
    }
}

pub struct CubeCountPlan {}

impl CubeCountPlan {
    pub fn resolve(&self) -> CubeCount {
        CubeCount::Static(1, 1, 1)
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
///
/// This enum is given as runtime input to the matmul
pub enum CubeCountInput {
    Tmp { dummy: u32 },
}
