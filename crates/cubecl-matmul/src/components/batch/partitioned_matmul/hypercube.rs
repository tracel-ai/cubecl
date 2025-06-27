use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

use crate::components::{MatmulProblem, TilingScheme};
use crate::kernels::MatmulSetupError;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct HypercubeConfig {
    cube_span: CubeSpan,
    pub global_order: GlobalOrder,
    pub cube_count_plan_config: CubeCountPlanConfig,
}

pub struct HypercubeConfigBuilder<'a> {
    tiling_scheme: &'a TilingScheme,
    global_order: GlobalOrderConfig,
    cube_count_plan_config: Option<CubeCountPlanConfig>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
// Number of elements each cube covers in the tensors
pub struct CubeSpan {
    m: u32,
    n: u32,
    batch: u32,
}

#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Describes the global traversal order as flattened cube position increases.
///
/// - `RowMajor`: standard row-first traversal
/// - `ColMajor`: standard column-first traversal
/// - `SwizzleColMajor(w)`: zigzag pattern down columns, with `w`-wide steps
/// - `SwizzleRowMajor(w)`: zigzag pattern across rows, with `w`-wide steps
///
/// Special cases:
/// - `SwizzleColMajor(1)` is equivalent to `ColMajor`
/// - `SwizzleRowMajor(1)` is equivalent to `RowMajor`
pub enum GlobalOrder {
    #[default]
    RowMajor,
    ColMajor,
    SwizzleRowMajor(u32),
    SwizzleColMajor(u32),
}

#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum CubeCountPlanConfig {
    /// X: num cubes in m, Y: num cubes in n, Z: num cubes in batch
    FromProblem,

    /// X: num SMs, Y: num cubes per SM
    SmFirst {
        num_sms: u32,
        sm_usage: SmAllocation,
    },

    /// X: num cubes per SM, Y: num SMs
    CubeFirst {
        num_sms: u32,
        sm_usage: SmAllocation,
    },

    #[default]
    /// X: total cubes flattened (num SMs * num cubes per SM)
    Flattened,
}

#[derive(CubeType, CubeLaunch)]
pub enum CubeCountPlan {
    FromProblem {
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
    },
    SmFirst {
        num_sms_used: u32,
        cubes_per_sm: u32,
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
    },
    CubeFirst {
        num_sms_used: u32,
        cubes_per_sm: u32,
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
    },
    Flattened {
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
    },
}

/// Controls how Streaming Multiprocessors (SMs) are assigned cubes.
///
/// - `Exact`: Balanced allocation using GCD (e.g., 120 cubes, 16 SMs → 4 SMs × 30 cubes)
/// - `Full`: Uses all SMs even if it overallocates (e.g., 120 cubes, 16 SMs → 16 SMs × 8 cubes = 128 total cubes)
/// - `Overallocate`: Allows extra SMs within a specified fraction (e.g., up to 25% overuse)
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum SmAllocation {
    /// Balanced: uses GCD to minimize slack.
    Exact,

    /// Uses all SMs, possibly overallocating total cubes.
    Full,

    /// Allows overallocating SMs up to a ratio.
    Ratio {
        max_extra_numerator: u32,
        max_extra_denominator: u32,
    },
}

impl GlobalOrder {
    /// Since they are equivalent but the latter form will skip some calculations,
    /// - `SwizzleColMajor(1)` becomes `ColMajor`
    /// - `SwizzleRowMajor(1)` becomes `RowMajor`
    pub fn canonicalize(self) -> Self {
        match self {
            GlobalOrder::SwizzleColMajor(1) => GlobalOrder::ColMajor,
            GlobalOrder::SwizzleRowMajor(1) => GlobalOrder::RowMajor,
            _ => self,
        }
    }
}

impl SmAllocation {
    fn allocate(&self, num_sms: u32, total_cubes: u32) -> (u32, u32) {
        match self {
            SmAllocation::Exact => SmAllocation::Ratio {
                max_extra_numerator: 0,
                max_extra_denominator: 1,
            }
            .allocate(num_sms, total_cubes),

            SmAllocation::Full => SmAllocation::Ratio {
                max_extra_numerator: u32::MAX,
                max_extra_denominator: 1,
            }
            .allocate(num_sms, total_cubes),

            SmAllocation::Ratio {
                max_extra_numerator,
                max_extra_denominator,
            } => {
                let max_slack = num_sms
                    .saturating_mul(*max_extra_numerator)
                    .div_ceil(*max_extra_denominator);

                let fallback_cubes_per_sm = total_cubes.div_ceil(num_sms);
                let mut best = (num_sms, fallback_cubes_per_sm);

                // Inline closure to generate divisors in descending order
                let divisors_desc = |n: u32| {
                    let mut divs = Vec::new();
                    let mut i = 1;

                    while i * i <= n {
                        if n % i == 0 {
                            divs.push(i);
                            if i != n / i {
                                divs.push(n / i);
                            }
                        }
                        i += 1;
                    }

                    divs.sort_by(|a, b| b.cmp(a)); // descending
                    divs.into_iter()
                };

                for sms_used in divisors_desc(num_sms) {
                    let cubes_per_sm = total_cubes.div_ceil(sms_used);
                    let total_allocated = cubes_per_sm * sms_used;
                    let slack = total_allocated.saturating_sub(total_cubes);

                    if slack <= max_slack {
                        best = (sms_used, cubes_per_sm);
                        break;
                    }
                }

                best
            }
        }
    }
}

impl CubeCountPlanConfig {
    pub fn can_yield_extra_cubes(&self) -> bool {
        match self {
            CubeCountPlanConfig::FromProblem | CubeCountPlanConfig::Flattened => false,
            CubeCountPlanConfig::SmFirst { .. } | CubeCountPlanConfig::CubeFirst { .. } => true,
        }
    }
}

impl HypercubeConfig {
    pub fn builder<'a>(tiling_scheme: &'a TilingScheme) -> HypercubeConfigBuilder<'a> {
        HypercubeConfigBuilder::new(tiling_scheme)
    }

    pub fn cube_count_plan(&self, problem: &MatmulProblem) -> CubeCountPlan {
        let m_cubes = (problem.m as u32).div_ceil(self.cube_span.m);
        let n_cubes = (problem.n as u32).div_ceil(self.cube_span.n);
        let batch_cubes = (problem.num_batches() as u32).div_ceil(self.cube_span.batch);

        match self.cube_count_plan_config {
            CubeCountPlanConfig::FromProblem => CubeCountPlan::FromProblem {
                m_cubes,
                n_cubes,
                batch_cubes,
            },
            CubeCountPlanConfig::SmFirst {
                num_sms,
                sm_usage: sms_partitioning,
            } => {
                let (num_sms_used, cubes_per_sm) =
                    sms_partitioning.allocate(num_sms, m_cubes * n_cubes * batch_cubes);
                CubeCountPlan::SmFirst {
                    num_sms_used,
                    cubes_per_sm,
                    m_cubes,
                    n_cubes,
                    batch_cubes,
                }
            }
            CubeCountPlanConfig::CubeFirst {
                num_sms,
                sm_usage: sms_partitioning,
            } => {
                let (num_sms_used, cubes_per_sm) =
                    sms_partitioning.allocate(num_sms, m_cubes * n_cubes * batch_cubes);
                CubeCountPlan::CubeFirst {
                    num_sms_used,
                    cubes_per_sm,
                    m_cubes,
                    n_cubes,
                    batch_cubes,
                }
            }
            CubeCountPlanConfig::Flattened => CubeCountPlan::Flattened {
                m_cubes,
                n_cubes,
                batch_cubes,
            },
        }
    }

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

impl CubeCountPlan {
    // Resolves the cube count plan into a concrete [`CubeCount`].
    pub fn resolve(&self) -> CubeCount {
        match self {
            CubeCountPlan::FromProblem {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => CubeCount::Static(*m_cubes, *n_cubes, *batch_cubes),
            CubeCountPlan::SmFirst {
                num_sms_used,
                cubes_per_sm,
                m_cubes: _,
                n_cubes: _,
                batch_cubes: _,
            } => CubeCount::Static(*num_sms_used, *cubes_per_sm, 1),
            CubeCountPlan::CubeFirst {
                num_sms_used,
                cubes_per_sm,
                m_cubes: _,
                n_cubes: _,
                batch_cubes: _,
            } => CubeCount::Static(*cubes_per_sm, *num_sms_used, 1),
            CubeCountPlan::Flattened {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => CubeCount::Static(*m_cubes * *n_cubes * *batch_cubes, 1, 1),
        }
    }

    pub fn to_args<'a, R: Runtime>(&self) -> CubeCountPlanArgs<'a, R> {
        match self {
            CubeCountPlan::FromProblem {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => CubeCountPlanArgs::FromProblem {
                m_cubes: ScalarArg::new(*m_cubes),
                n_cubes: ScalarArg::new(*n_cubes),
                batch_cubes: ScalarArg::new(*batch_cubes),
            },
            CubeCountPlan::SmFirst {
                num_sms_used,
                cubes_per_sm,
                m_cubes,
                n_cubes,
                batch_cubes,
            } => CubeCountPlanArgs::SmFirst {
                num_sms_used: ScalarArg::new(*num_sms_used),
                cubes_per_sm: ScalarArg::new(*cubes_per_sm),
                m_cubes: ScalarArg::new(*m_cubes),
                n_cubes: ScalarArg::new(*n_cubes),
                batch_cubes: ScalarArg::new(*batch_cubes),
            },
            CubeCountPlan::CubeFirst {
                num_sms_used,
                cubes_per_sm,
                m_cubes,
                n_cubes,
                batch_cubes,
            } => CubeCountPlanArgs::CubeFirst {
                num_sms_used: ScalarArg::new(*num_sms_used),
                cubes_per_sm: ScalarArg::new(*cubes_per_sm),
                m_cubes: ScalarArg::new(*m_cubes),
                n_cubes: ScalarArg::new(*n_cubes),
                batch_cubes: ScalarArg::new(*batch_cubes),
            },
            CubeCountPlan::Flattened {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => CubeCountPlanArgs::Flattened {
                m_cubes: ScalarArg::new(*m_cubes),
                n_cubes: ScalarArg::new(*n_cubes),
                batch_cubes: ScalarArg::new(*batch_cubes),
            },
        }
    }
}

#[cube]
impl CubeCountPlan {
    pub fn max_cube_pos(&self) -> u32 {
        match self {
            CubeCountPlan::FromProblem {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => *m_cubes * *n_cubes * *batch_cubes,
            CubeCountPlan::SmFirst {
                num_sms_used: _,
                cubes_per_sm: _,
                m_cubes,
                n_cubes,
                batch_cubes,
            } => *m_cubes * *n_cubes * *batch_cubes,
            CubeCountPlan::CubeFirst {
                num_sms_used: _,
                cubes_per_sm: _,
                m_cubes,
                n_cubes,
                batch_cubes,
            } => *m_cubes * *n_cubes * *batch_cubes,
            CubeCountPlan::Flattened {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => *m_cubes * *n_cubes * *batch_cubes,
        }
    }

    /// Given a cube position (SM ID, local index), returns the tensor coordinates (m, n, batch).
    pub fn cube_pos_to_tensor_pos(&self, #[comptime] global_order: GlobalOrder) -> (u32, u32, u32) {
        match self {
            CubeCountPlan::FromProblem {
                m_cubes: _,
                n_cubes: _,
                batch_cubes: _,
            } => (CUBE_POS_X, CUBE_POS_Y, CUBE_POS_Z),
            CubeCountPlan::SmFirst {
                num_sms_used: _,
                cubes_per_sm: _,
                m_cubes,
                n_cubes,
                batch_cubes: _,
            } => self.absolute_index_to_m_n_batch(CUBE_POS, *m_cubes, *n_cubes, global_order),
            CubeCountPlan::CubeFirst {
                num_sms_used: _,
                cubes_per_sm,
                m_cubes,
                n_cubes,
                batch_cubes: _,
            } => self.absolute_index_to_m_n_batch(
                CUBE_POS_Y * cubes_per_sm + CUBE_POS_X,
                *m_cubes,
                *n_cubes,
                global_order,
            ),
            CubeCountPlan::Flattened {
                m_cubes,
                n_cubes,
                batch_cubes: _,
            } => self.absolute_index_to_m_n_batch(CUBE_POS_X, *m_cubes, *n_cubes, global_order),
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

#[derive(Default)]
/// Used to create [GlobalOrder].
#[allow(unused)]
pub enum GlobalOrderConfig {
    /// It creates the default global order.
    #[default]
    Default,
    /// Creates swizzle row global order if possible.
    ///
    /// Fallbacks to row global order otherwise.
    SwizzleRow { m: u32, w: u32 },
    /// Creates swizzle col global order if possible.
    ///
    /// Fallbacks to col global order otherwise.
    SwizzleCol { n: u32, w: u32 },
}

impl GlobalOrderConfig {
    pub fn into_order(self, span: &CubeSpan) -> GlobalOrder {
        match self {
            GlobalOrderConfig::Default => GlobalOrder::default(),
            GlobalOrderConfig::SwizzleRow { m, w } => {
                let m_cubes = m.div_ceil(span.m);
                if m_cubes % w != 0 {
                    GlobalOrder::RowMajor
                } else {
                    GlobalOrder::SwizzleRowMajor(w)
                }
            }
            GlobalOrderConfig::SwizzleCol { n, w } => {
                let n_cubes = n.div_ceil(span.n);
                if n_cubes % w != 0 {
                    GlobalOrder::RowMajor
                } else {
                    GlobalOrder::SwizzleRowMajor(w)
                }
            }
        }
        .canonicalize()
    }
}

impl<'a> HypercubeConfigBuilder<'a> {
    fn new(tiling_scheme: &'a TilingScheme) -> Self {
        Self {
            tiling_scheme,
            global_order: GlobalOrderConfig::default(),
            cube_count_plan_config: None,
        }
    }

    pub fn global_order(mut self, global_order: GlobalOrderConfig) -> Self {
        self.global_order = global_order;
        self
    }

    pub fn cube_count_plan(mut self, cube_count_plan_config: CubeCountPlanConfig) -> Self {
        self.cube_count_plan_config = Some(cube_count_plan_config);
        self
    }

    pub fn build(self) -> HypercubeConfig {
        let cube_span = CubeSpan {
            m: self.tiling_scheme.elements_in_global_partition_m(),
            n: self.tiling_scheme.elements_in_global_partition_n(),
            batch: self.tiling_scheme.global_partition_size.batches,
        };

        let global_order = self.global_order.into_order(&cube_span);
        let cube_pos_strategy = self.cube_count_plan_config.unwrap_or_default();

        HypercubeConfig {
            cube_span,
            global_order,
            cube_count_plan_config: cube_pos_strategy,
        }
    }
}

#[cube]
/// Maps a linear `index` to 2D zigzag coordinates `(x, y)` within horizontal or vertical strips.
///
/// Each strip is made of `num_steps` steps, each of length `step_length`.
/// Strips alternate direction: even strips go top-down, odd strips bottom-up.
/// Steps alternate direction: even steps go left-to-right, odd steps right-to-left.
///
/// - Prefer **odd `num_steps`** for smoother transitions between strips.
/// - Prefer **power-of-two `step_length`** for better performance.
///
/// # Parameters
/// - `index`: linear input index
/// - `num_steps`: number of snaking steps in a strip
/// - `step_length`: number of elements in each step (must be > 0)
///
/// # Returns
/// `(x, y)` coordinates after swizzling
fn swizzle(index: u32, num_steps: u32, #[comptime] step_length: u32) -> (u32, u32) {
    comptime!(assert!(step_length > 0));

    let num_elements_per_strip = num_steps * step_length;
    let strip_index = index / num_elements_per_strip;
    let pos_in_strip = index % num_elements_per_strip;
    let strip_offset = step_length * strip_index;

    // Indices without regards to direction
    let abs_step_index = pos_in_strip / step_length;
    let abs_pos_in_step = pos_in_strip % step_length;

    // Top-down (0) or Bottom-up (1)
    let strip_direction = strip_index % 2;
    // Left-right (0) or Right-left (1)
    let step_direction = abs_step_index % 2;

    // Update indices with direction
    let step_index =
        strip_direction * (num_steps - abs_step_index - 1) + (1 - strip_direction) * abs_step_index;

    let pos_in_step = if comptime!(step_length & (step_length - 1) == 0) {
        abs_pos_in_step ^ (step_direction * (step_length - 1))
    } else {
        step_direction * (step_length - abs_pos_in_step - 1)
            + (1 - step_direction) * abs_pos_in_step
    };

    (step_index, pos_in_step + strip_offset)
}
