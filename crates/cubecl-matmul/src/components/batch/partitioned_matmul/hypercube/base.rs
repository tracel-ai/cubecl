use cubecl_core::CubeCount;

use crate::components::{
    MatmulProblem, MatmulSetupError, TilingScheme,
    batch::partitioned_matmul::hypercube::{
        cube_count_plan::{CubeCountPlan, CubeCountPlanConfig, CubeCountPlanSelection},
        global_order::{GlobalOrder, GlobalOrderSelection},
    },
};

#[derive(Debug, Clone)]
pub struct HypercubeSelection {
    pub cube_span: CubeSpan,
    pub global_order: GlobalOrder,
    pub cube_count_plan_selection: CubeCountPlanSelection,
}

pub struct HypercubeSelectionBuilder<'a> {
    tiling_scheme: &'a TilingScheme,
    global_order: GlobalOrderSelection,
    cube_count_plan_config: Option<CubeCountPlanSelection>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct HypercubeConfig {
    pub cube_span: CubeSpan,
    pub global_order: GlobalOrder,
    pub cube_count_plan_config: CubeCountPlanConfig,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
// Number of elements each cube covers in the tensors
pub struct CubeSpan {
    pub m: u32,
    pub n: u32,
    pub batch: u32,
}

impl HypercubeSelection {
    pub fn builder<'a>(tiling_scheme: &'a TilingScheme) -> HypercubeSelectionBuilder<'a> {
        HypercubeSelectionBuilder::new(tiling_scheme)
    }

    pub(crate) fn to_hypercube_config(
        &self,
        problem: &MatmulProblem,
        max_cube_count: CubeCount,
    ) -> HypercubeConfig {
        let cube_count_plan = CubeCountPlan::from_selection(self, problem, max_cube_count);
        let cube_count_plan_config = CubeCountPlanConfig::from_cube_count_plan(cube_count_plan);

        HypercubeConfig {
            cube_span: self.cube_span,
            global_order: self.global_order,
            cube_count_plan_config,
        }
    }
}

impl HypercubeConfig {
    pub fn validate(&self, problem: &MatmulProblem) -> Result<(), MatmulSetupError> {
        let m_cubes = (problem.m as u32).div_ceil(self.cube_span.m);
        let n_cubes = (problem.n as u32).div_ceil(self.cube_span.n);

        use GlobalOrder::*;

        match self.global_order {
            RowMajor | ColMajor => Ok(()),

            SwizzleRowMajor(w) if m_cubes % w != 0 => {
                Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                    "In swizzle row major, number of cubes in m {m_cubes:?} must be divisible by swizzle step length {w:?}."
                ))))
            }

            SwizzleColMajor(w) if n_cubes % w != 0 => {
                Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                    "In swizzle col major, number of cubes in n {n_cubes:?} must be divisible by swizzle step length {w:?}."
                ))))
            }

            _ => Ok(()),
        }
    }
}

impl<'a> HypercubeSelectionBuilder<'a> {
    fn new(tiling_scheme: &'a TilingScheme) -> Self {
        Self {
            tiling_scheme,
            global_order: GlobalOrderSelection::default(),
            cube_count_plan_config: None,
        }
    }

    pub fn global_order(mut self, global_order: GlobalOrderSelection) -> Self {
        self.global_order = global_order;
        self
    }

    pub fn cube_count_plan(mut self, cube_count_plan_config: CubeCountPlanSelection) -> Self {
        self.cube_count_plan_config = Some(cube_count_plan_config);
        self
    }

    pub fn build(self) -> HypercubeSelection {
        let cube_span = CubeSpan {
            m: self.tiling_scheme.elements_in_global_partition_m(),
            n: self.tiling_scheme.elements_in_global_partition_n(),
            batch: self.tiling_scheme.global_partition_size.batches,
        };

        let global_order = self.global_order.into_order(&cube_span);
        let cube_pos_strategy = self.cube_count_plan_config.unwrap_or_default();

        HypercubeSelection {
            cube_span,
            global_order,
            cube_count_plan_selection: cube_pos_strategy,
        }
    }
}

impl HypercubeConfig {
    pub fn cube_count_plan(
        &self,
        problem: &MatmulProblem,
        max_cube_count: CubeCount,
    ) -> CubeCountPlan {
        CubeCountPlan::from_config(self, problem, max_cube_count)
    }
}
