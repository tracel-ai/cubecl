use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::components::MatmulProblem;
use crate::components::batch::partitioned_matmul::hypercube::global_order::{GlobalOrder, swizzle};
use crate::components::batch::partitioned_matmul::hypercube::sm_allocation::SmAllocation;
use crate::components::batch::{HypercubeConfig, HypercubeSelection};

#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Front-facing configuration when crafting a MatmulSelection
/// Allows choosing a strategy before knowing actual values
pub enum CubeCountPlanSelection {
    #[default]
    /// X: num cubes in m, Y: num cubes in n, Z: num cubes in batch
    FromProblem,

    /// If not cubes_first: X: num SMs, Y: num cubes per SM
    /// If cubes_first: X: num cubes per SM, Y: num SMs
    Sm {
        cubes_first: bool,
        num_sms: u32,
        sm_usage: SmAllocation,
    },

    /// X: total cubes flattened (num SMs * num cubes per SM)
    Flattened,

    /// Heuristically find a balance for X, Y, Z that respects hardware limits
    Spread,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Informations necessary in the computation of the CubeCount.
/// Because this struct depends on the problem size, it is simplified into
/// [CubeCountPlanConfig] to be injected as comptime in the kernel.
/// 
/// Refer to [CubeCountPlanSelection] for more details
pub enum CubeCountPlan {
    FromProblem {
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
    },
    Sm {
        cubes_first: bool,
        num_sms_used: u32,
        cubes_per_sm: u32,
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
        num_sms: u32,
        sm_usage: SmAllocation,
    },
    Flattened {
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
    },
    Spread {
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
        x: u32,
        y: u32,
        z: u32,
    },
}

impl CubeCountPlan {
    /// Whether the CubeCount will have more cubes than strictly necessary.
    pub fn can_yield_extra_cubes(&self) -> bool {
        match self {
            CubeCountPlan::FromProblem { .. } | CubeCountPlan::Flattened { .. } => false,
            CubeCountPlan::Sm {
                num_sms_used,
                cubes_per_sm,
                m_cubes,
                n_cubes,
                batch_cubes,
                ..
            } => num_sms_used * cubes_per_sm != m_cubes * n_cubes * batch_cubes,
            CubeCountPlan::Spread {
                m_cubes,
                n_cubes,
                batch_cubes,
                x,
                y,
                z,
            } => m_cubes * n_cubes * batch_cubes != x * y * z,
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Config derived from CubeCountPlan to be used comptime in kernels
///
/// Refer to [CubeCountPlanSelection] for more details
pub enum CubeCountPlanConfig {
    FromProblem,

    Sm {
        cubes_first: bool,
        num_sms: u32,
        sm_usage: SmAllocation,
        can_yield_extra_cubes: bool,
    },

    Flattened,

    Spread {
        can_yield_extra_cubes: bool,
    },
}

#[derive(CubeType, CubeLaunch)]
/// CubeCountPlan stripped of non-essential runtime information
///
/// This enum is given as runtime input to the matmul
pub enum CubeCountInput {
    FromProblem,
    SmFirst {
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
    },
    CubeFirst {
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
    },
    Flattened {
        m_cubes: u32,
        n_cubes: u32,
    },
    Spread {
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
    },
}

impl CubeCountPlan {
    // Will check if the wanted cube count plan is possible, otherwise will fallback to spread
    pub fn from_selection(
        selection: &HypercubeSelection,
        problem: &MatmulProblem,
        max_cube_count: CubeCount,
    ) -> CubeCountPlan {
        let (max_x, max_y, max_z) = match max_cube_count {
            CubeCount::Static(x, y, z) => (x, y, z),
            CubeCount::Dynamic(_) => panic!("Dynamic cube count not supported for cube count plan"),
        };

        let m_cubes = (problem.m as u32).div_ceil(selection.cube_span.m);
        let n_cubes = (problem.n as u32).div_ceil(selection.cube_span.n);
        let batch_cubes = (problem.num_batches() as u32).div_ceil(selection.cube_span.batch);

        let plan = match selection.cube_count_plan_selection {
            CubeCountPlanSelection::FromProblem => {
                if m_cubes > max_x || n_cubes > max_y || batch_cubes > max_z {
                    None
                } else {
                    Some(CubeCountPlan::FromProblem {
                        m_cubes,
                        n_cubes,
                        batch_cubes,
                    })
                }
            }
            CubeCountPlanSelection::Sm {
                cubes_first,
                num_sms,
                sm_usage,
            } => {
                let (num_sms_used, cubes_per_sm) =
                    sm_usage.allocate(num_sms, m_cubes * n_cubes * batch_cubes);

                if (cubes_per_sm >= if cubes_first { max_x } else { max_y })
                    || (num_sms_used >= if cubes_first { max_y } else { max_x })
                {
                    None
                } else {
                    Some(CubeCountPlan::Sm {
                        cubes_first,
                        num_sms_used,
                        cubes_per_sm,
                        m_cubes,
                        n_cubes,
                        batch_cubes,
                        num_sms,
                        sm_usage,
                    })
                }
            }
            CubeCountPlanSelection::Flattened => {
                if m_cubes * n_cubes * batch_cubes >= max_x {
                    None
                } else {
                    Some(CubeCountPlan::Flattened {
                        m_cubes,
                        n_cubes,
                        batch_cubes,
                    })
                }
            }
            CubeCountPlanSelection::Spread => None,
        };

        plan.unwrap_or_else(|| {
            spread_cube_count_plan(m_cubes, n_cubes, batch_cubes, max_x, max_y, max_z)
        })
    }

    /// Because we don't want to store the CubeCountPlan values in config, we have to recompute it
    ///
    /// Assumes the hypercube config is valid
    pub fn from_config(
        config: &HypercubeConfig,
        problem: &MatmulProblem,
        max_cube_count: CubeCount,
    ) -> CubeCountPlan {
        let (max_x, max_y, max_z) = match max_cube_count {
            CubeCount::Static(x, y, z) => (x, y, z),
            CubeCount::Dynamic(_) => panic!("Dynamic cube count not supported for cube count plan"),
        };

        let m_cubes = (problem.m as u32).div_ceil(config.cube_span.m);
        let n_cubes = (problem.n as u32).div_ceil(config.cube_span.n);
        let batch_cubes = (problem.num_batches() as u32).div_ceil(config.cube_span.batch);

        match config.cube_count_plan_config {
            CubeCountPlanConfig::FromProblem => CubeCountPlan::FromProblem {
                m_cubes,
                n_cubes,
                batch_cubes,
            },
            CubeCountPlanConfig::Sm {
                cubes_first,
                num_sms,
                sm_usage,
                ..
            } => {
                let (num_sms_used, cubes_per_sm) =
                    sm_usage.allocate(num_sms, m_cubes * n_cubes * batch_cubes);
                CubeCountPlan::Sm {
                    cubes_first,
                    num_sms_used,
                    cubes_per_sm,
                    m_cubes,
                    n_cubes,
                    batch_cubes,
                    num_sms,
                    sm_usage,
                }
            }
            CubeCountPlanConfig::Flattened => CubeCountPlan::Flattened {
                m_cubes,
                n_cubes,
                batch_cubes,
            },
            CubeCountPlanConfig::Spread { .. } => {
                spread_cube_count_plan(m_cubes, n_cubes, batch_cubes, max_x, max_y, max_z)
            }
        }
    }
}

impl CubeCountPlanConfig {
    /// Whether the CubeCount will have more cubes than strictly necessary.
    pub fn can_yield_extra_cubes(&self) -> bool {
        match self {
            CubeCountPlanConfig::FromProblem | CubeCountPlanConfig::Flattened => false,
            CubeCountPlanConfig::Sm {
                can_yield_extra_cubes,
                ..
            } => *can_yield_extra_cubes,
            CubeCountPlanConfig::Spread {
                can_yield_extra_cubes,
            } => *can_yield_extra_cubes,
        }
    }

    pub(crate) fn from_cube_count_plan(cube_count_plan: CubeCountPlan) -> CubeCountPlanConfig {
        match cube_count_plan {
            CubeCountPlan::FromProblem { .. } => CubeCountPlanConfig::FromProblem,
            CubeCountPlan::Sm {
                cubes_first,
                num_sms,
                sm_usage,
                ..
            } => CubeCountPlanConfig::Sm {
                cubes_first,
                num_sms,
                sm_usage,
                can_yield_extra_cubes: cube_count_plan.can_yield_extra_cubes(),
            },
            CubeCountPlan::Flattened { .. } => CubeCountPlanConfig::Flattened,
            CubeCountPlan::Spread { .. } => CubeCountPlanConfig::Spread {
                can_yield_extra_cubes: cube_count_plan.can_yield_extra_cubes(),
            },
        }
    }
}

/// Heuristic algorithm to factor the total number of cubes into (x, y, z) dimensions
/// such that no dimension surpasses its maximum.
pub(crate) fn spread_cube_count_plan(
    m_cubes: u32,
    n_cubes: u32,
    batch_cubes: u32,
    max_x: u32,
    max_y: u32,
    max_z: u32,
) -> CubeCountPlan {
    let total_cubes = m_cubes * n_cubes * batch_cubes;

    let mut best = None;

    let mut z = max_z;
    while z >= 1 {
        let xy_cubes = total_cubes.div_ceil(z);

        let mut y = max_y;
        while y >= 1 {
            let x = xy_cubes.div_ceil(y);
            if x <= max_x {
                let volume = x * y * z;
                let score = (volume, std::cmp::Reverse(z), std::cmp::Reverse(y));

                if best.is_none_or(|(_, _, _, _, best_score)| score < best_score) {
                    best = Some((x, y, z, volume, score));
                }
            }

            if y == 1 {
                break;
            }
            y /= 2;
        }

        if z == 1 {
            break;
        }
        z /= 2;
    }

    if let Some((x, y, z, _, _)) = best {
        CubeCountPlan::Spread {
            m_cubes,
            n_cubes,
            batch_cubes,
            x,
            y,
            z,
        }
    } else {
        panic!("No valid cube spread plan")
    }
}

impl CubeCountPlan {
    // Resolves the cube count plan into a concrete [`CubeCount`].
    pub fn resolve(&self) -> CubeCount {
        match self {
            CubeCountPlan::FromProblem {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => CubeCount::Static(*m_cubes, *n_cubes, *batch_cubes),
            CubeCountPlan::Sm {
                cubes_first,
                num_sms_used,
                cubes_per_sm,
                ..
            } => match cubes_first {
                true => CubeCount::Static(*cubes_per_sm, *num_sms_used, 1),
                false => CubeCount::Static(*num_sms_used, *cubes_per_sm, 1),
            },
            CubeCountPlan::Flattened {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => CubeCount::Static(*m_cubes * *n_cubes * *batch_cubes, 1, 1),
            CubeCountPlan::Spread { x, y, z, .. } => CubeCount::Static(*x, *y, *z),
        }
    }

    /// Make a CubeCountInput from CubeCountPlan
    pub fn as_args<'a, R: Runtime>(&self) -> CubeCountInputArgs<'a, R> {
        match self {
            CubeCountPlan::FromProblem { .. } => CubeCountInputArgs::FromProblem,
            CubeCountPlan::Sm {
                cubes_first,
                m_cubes,
                n_cubes,
                batch_cubes,
                ..
            } => match cubes_first {
                true => CubeCountInputArgs::CubeFirst {
                    m_cubes: ScalarArg::new(*m_cubes),
                    n_cubes: ScalarArg::new(*n_cubes),
                    batch_cubes: ScalarArg::new(*batch_cubes),
                },
                false => CubeCountInputArgs::SmFirst {
                    m_cubes: ScalarArg::new(*m_cubes),
                    n_cubes: ScalarArg::new(*n_cubes),
                    batch_cubes: ScalarArg::new(*batch_cubes),
                },
            },
            CubeCountPlan::Flattened {
                m_cubes, n_cubes, ..
            } => CubeCountInputArgs::Flattened {
                m_cubes: ScalarArg::new(*m_cubes),
                n_cubes: ScalarArg::new(*n_cubes),
            },
            CubeCountPlan::Spread {
                m_cubes,
                n_cubes,
                batch_cubes,
                ..
            } => CubeCountInputArgs::Spread {
                m_cubes: ScalarArg::new(*m_cubes),
                n_cubes: ScalarArg::new(*n_cubes),
                batch_cubes: ScalarArg::new(*batch_cubes),
            },
        }
    }
}

#[cube]
impl CubeCountInput {
    /// Returns the number of valid cubes
    pub fn num_valid_cubes(&self) -> u32 {
        match self {
            CubeCountInput::FromProblem | CubeCountInput::Flattened { .. } => {
                panic!("Shouldn't need to be called because the cube count should always be exact")
            }
            CubeCountInput::SmFirst {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => *m_cubes * *n_cubes * *batch_cubes,
            CubeCountInput::CubeFirst {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => *m_cubes * *n_cubes * *batch_cubes,
            CubeCountInput::Spread {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => *m_cubes * *n_cubes * *batch_cubes,
        }
    }

    /// Given a cube position (SM ID, local index), returns the tensor coordinates (m, n, batch).
    pub fn cube_pos_to_tensor_pos(&self, #[comptime] global_order: GlobalOrder) -> (u32, u32, u32) {
        match self {
            CubeCountInput::FromProblem => (CUBE_POS_X, CUBE_POS_Y, CUBE_POS_Z),
            CubeCountInput::SmFirst {
                m_cubes, n_cubes, ..
            } => self.absolute_index_to_m_n_batch(CUBE_POS, *m_cubes, *n_cubes, global_order),
            CubeCountInput::CubeFirst {
                m_cubes, n_cubes, ..
            } => self.absolute_index_to_m_n_batch(
                CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X,
                *m_cubes,
                *n_cubes,
                global_order,
            ),
            CubeCountInput::Flattened { m_cubes, n_cubes } => {
                self.absolute_index_to_m_n_batch(CUBE_POS_X, *m_cubes, *n_cubes, global_order)
            }
            CubeCountInput::Spread {
                m_cubes, n_cubes, ..
            } => self.absolute_index_to_m_n_batch(CUBE_POS, *m_cubes, *n_cubes, global_order),
        }
    }

    fn absolute_index_to_m_n_batch(
        &self,
        absolute_index: u32,
        m_cubes: u32,
        n_cubes: u32,
        #[comptime] global_order: GlobalOrder,
    ) -> (u32, u32, u32) {
        let batch_stride = m_cubes * n_cubes;
        let batch_pos = absolute_index / batch_stride;
        let matrix_pos = absolute_index % batch_stride;

        let (m_pos, n_pos) = match comptime!(global_order) {
            GlobalOrder::RowMajor => (matrix_pos / n_cubes, matrix_pos % n_cubes),
            GlobalOrder::ColMajor => (matrix_pos % m_cubes, matrix_pos / m_cubes),
            GlobalOrder::SwizzleRowMajor(w) => {
                let (x, y) = swizzle(matrix_pos, n_cubes, w);
                (y, x)
            }
            GlobalOrder::SwizzleColMajor(w) => swizzle(matrix_pos, m_cubes, w),
        };

        (m_pos, n_pos, batch_pos)
    }
}
