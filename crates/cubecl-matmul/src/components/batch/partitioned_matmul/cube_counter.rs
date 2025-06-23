use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

use crate::components::{MatmulProblem, TilingScheme};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
// TODO rename both name and variants (row/col major?)
pub enum GlobalPartitioning {
    Natural,
    Transposed,
}

#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum CubeCountStrategyConfig {
    /// X: num cubes in m, Y: num cubes in n, Z: num cubes in batch
    FromProblem,

    /// X: num SMs, Y: num cubes per SM
    /// Given value: num SMs
    SmPerCubeFirst {
        num_sms: u32,
        sms_partitioning: SmsCubePartitioning,
    },

    /// X: num cubes per SM, Y: num SMs
    /// Given value: num SMs
    CubePerSmFirst {
        num_sms: u32,
        sms_partitioning: SmsCubePartitioning,
    },

    #[default]
    /// X: total cubes flattened (num SMs * num cubes per SM)
    Flat,
}

#[derive(CubeType, CubeLaunch)]
pub enum CubeCountStrategy {
    FromProblem {
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
    },
    SmPerCubeFirst {
        num_sms_used: u32,
        cubes_per_sm: u32,
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
    },
    CubePerSmFirst {
        num_sms_used: u32,
        cubes_per_sm: u32,
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
    },
    Flat {
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
    },
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum SmsCubePartitioning {
    /// Uses the exact GCD of total_cubes and num_sms to balance perfectly
    /// Equal to Heuristic with 0% slack
    ExactGcd,

    /// Forces using all num_sms, even if it causes excess cubes
    /// Equal to Heuristic with infinite% slack
    ForceAllSms,

    /// Tries to find a divisor of `num_sms` such that the number of extra cubes
    /// (allocated but unused) does not exceed a given fraction of `num_sms`.
    ///
    /// The maximum tolerated slack is:
    ///     ceil((max_slack_numerator / max_slack_denominator) × num_sms)
    ///
    /// The heuristic chooses the smallest valid `sms_used` (i.e., a divisor of `num_sms`)
    /// such that:
    ///     sms_used × ceil(total_cubes / sms_used) - total_cubes <= max_slack
    ///
    /// Example:
    /// - total_cubes = 50
    /// - num_sms = 24
    /// - max_slack_numerator = 1
    /// - max_slack_denominator = 4  // → max_slack = ceil(24 × 1/4) = 6
    ///
    /// Then the heuristic might pick `sms_used = 6` and `cubes_per_sm = 9`
    /// yielding 54 cubes total (slack = 4).
    Heuristic {
        max_slack_numerator: u32,
        max_slack_denominator: u32,
    },
}

impl SmsCubePartitioning {
    fn partition_cubes(&self, num_sms: u32, total_cubes: u32) -> (u32, u32) {
        match self {
            SmsCubePartitioning::ExactGcd => SmsCubePartitioning::Heuristic {
                max_slack_numerator: 0,
                max_slack_denominator: 1,
            }
            .partition_cubes(num_sms, total_cubes),

            SmsCubePartitioning::ForceAllSms => SmsCubePartitioning::Heuristic {
                max_slack_numerator: u32::MAX,
                max_slack_denominator: 1,
            }
            .partition_cubes(num_sms, total_cubes),

            SmsCubePartitioning::Heuristic {
                max_slack_numerator,
                max_slack_denominator,
            } => {
                let max_slack = num_sms
                    .saturating_mul(*max_slack_numerator)
                    .div_ceil(*max_slack_denominator);

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

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
// What we can know comptime
pub struct CubeCounterConfig {
    cube_span: CubeSpan,
    pub global_partitioning: GlobalPartitioning,
    cube_pos_strategy: CubeCountStrategyConfig,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct CubeSpan {
    m: u32,
    n: u32,
    batch: u32,
}

impl CubeCounterConfig {
    pub fn new(
        tiling_scheme: &TilingScheme,
        global_partitioning: GlobalPartitioning,
        cube_pos_strategy: CubeCountStrategyConfig,
    ) -> Self {
        let cube_span = CubeSpan {
            m: tiling_scheme.elements_in_global_partition_m(),
            n: tiling_scheme.elements_in_global_partition_n(),
            batch: tiling_scheme.global_partition_size.batches,
        };

        Self {
            cube_span,
            global_partitioning,
            cube_pos_strategy,
        }
    }

    pub fn cube_count_data(&self, problem: &MatmulProblem) -> CubeCountStrategy {
        let m_cubes = (problem.m as u32).div_ceil(self.cube_span.m);
        let n_cubes = (problem.n as u32).div_ceil(self.cube_span.n);
        let batch_cubes = (problem.num_batches() as u32).div_ceil(self.cube_span.batch);

        match self.cube_pos_strategy {
            CubeCountStrategyConfig::FromProblem => CubeCountStrategy::FromProblem {
                m_cubes,
                n_cubes,
                batch_cubes,
            },
            CubeCountStrategyConfig::SmPerCubeFirst {
                num_sms,
                sms_partitioning,
            } => {
                let (num_sms_used, cubes_per_sm) =
                    sms_partitioning.partition_cubes(num_sms, m_cubes * n_cubes * batch_cubes);
                CubeCountStrategy::SmPerCubeFirst {
                    num_sms_used,
                    cubes_per_sm,
                    m_cubes,
                    n_cubes,
                    batch_cubes,
                }
            }
            CubeCountStrategyConfig::CubePerSmFirst {
                num_sms,
                sms_partitioning,
            } => {
                let (num_sms_used, cubes_per_sm) =
                    sms_partitioning.partition_cubes(num_sms, m_cubes * n_cubes * batch_cubes);
                CubeCountStrategy::CubePerSmFirst {
                    num_sms_used,
                    cubes_per_sm,
                    m_cubes,
                    n_cubes,
                    batch_cubes,
                }
            }
            CubeCountStrategyConfig::Flat => CubeCountStrategy::Flat {
                m_cubes,
                n_cubes,
                batch_cubes,
            },
        }
    }
}

impl CubeCountStrategy {
    pub fn to_cube_count(&self) -> CubeCount {
        match self {
            CubeCountStrategy::FromProblem {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => CubeCount::Static(*m_cubes, *n_cubes, *batch_cubes),
            CubeCountStrategy::SmPerCubeFirst {
                num_sms_used,
                cubes_per_sm,
                m_cubes: _,
                n_cubes: _,
                batch_cubes: _,
            } => CubeCount::Static(*num_sms_used, *cubes_per_sm, 1),
            CubeCountStrategy::CubePerSmFirst {
                num_sms_used,
                cubes_per_sm,
                m_cubes: _,
                n_cubes: _,
                batch_cubes: _,
            } => CubeCount::Static(*cubes_per_sm, *num_sms_used, 1),
            CubeCountStrategy::Flat {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => CubeCount::Static(*m_cubes * *n_cubes * *batch_cubes, 1, 1),
        }
    }

    pub fn to_args<'a, R: Runtime>(&self) -> CubeCountStrategyArgs<'a, R> {
        match self {
            CubeCountStrategy::FromProblem {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => CubeCountStrategyArgs::FromProblem {
                m_cubes: ScalarArg::new(*m_cubes),
                n_cubes: ScalarArg::new(*n_cubes),
                batch_cubes: ScalarArg::new(*batch_cubes),
            },
            CubeCountStrategy::SmPerCubeFirst {
                num_sms_used,
                cubes_per_sm,
                m_cubes,
                n_cubes,
                batch_cubes,
            } => CubeCountStrategyArgs::SmPerCubeFirst {
                num_sms_used: ScalarArg::new(*num_sms_used),
                cubes_per_sm: ScalarArg::new(*cubes_per_sm),
                m_cubes: ScalarArg::new(*m_cubes),
                n_cubes: ScalarArg::new(*n_cubes),
                batch_cubes: ScalarArg::new(*batch_cubes),
            },
            CubeCountStrategy::CubePerSmFirst {
                num_sms_used,
                cubes_per_sm,
                m_cubes,
                n_cubes,
                batch_cubes,
            } => CubeCountStrategyArgs::CubePerSmFirst {
                num_sms_used: ScalarArg::new(*num_sms_used),
                cubes_per_sm: ScalarArg::new(*cubes_per_sm),
                m_cubes: ScalarArg::new(*m_cubes),
                n_cubes: ScalarArg::new(*n_cubes),
                batch_cubes: ScalarArg::new(*batch_cubes),
            },
            CubeCountStrategy::Flat {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => CubeCountStrategyArgs::Flat {
                m_cubes: ScalarArg::new(*m_cubes),
                n_cubes: ScalarArg::new(*n_cubes),
                batch_cubes: ScalarArg::new(*batch_cubes),
            },
        }
    }
}

#[cube]
impl CubeCountStrategy {
    pub fn max_cube_pos(&self) -> u32 {
        match self {
            CubeCountStrategy::FromProblem {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => *m_cubes * *n_cubes * *batch_cubes,
            CubeCountStrategy::SmPerCubeFirst {
                num_sms_used: _,
                cubes_per_sm: _,
                m_cubes,
                n_cubes,
                batch_cubes,
            } => *m_cubes * *n_cubes * *batch_cubes,
            CubeCountStrategy::CubePerSmFirst {
                num_sms_used: _,
                cubes_per_sm: _,
                m_cubes,
                n_cubes,
                batch_cubes,
            } => *m_cubes * *n_cubes * *batch_cubes,
            CubeCountStrategy::Flat {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => *m_cubes * *n_cubes * *batch_cubes,
        }
    }

    /// Given a cube position (SM ID, local index), returns the tensor coordinates (m, n, batch).
    pub fn cube_pos_to_tensor_pos(
        &self,
        #[comptime] global_partitioning: GlobalPartitioning,
    ) -> (u32, u32, u32) {
        match self {
            CubeCountStrategy::FromProblem {
                m_cubes: _,
                n_cubes: _,
                batch_cubes: _,
            } => (CUBE_POS_X, CUBE_POS_Y, CUBE_POS_Z),
            CubeCountStrategy::SmPerCubeFirst {
                num_sms_used: _,
                cubes_per_sm: _,
                m_cubes,
                n_cubes,
                batch_cubes: _,
            } => {
                self.absolute_index_to_m_n_batch(CUBE_POS, *m_cubes, *n_cubes, global_partitioning)
            }
            CubeCountStrategy::CubePerSmFirst {
                num_sms_used: _,
                cubes_per_sm,
                m_cubes,
                n_cubes,
                batch_cubes: _,
            } => self.absolute_index_to_m_n_batch(
                CUBE_POS_Y * cubes_per_sm + CUBE_POS_X,
                *m_cubes,
                *n_cubes,
                global_partitioning,
            ),
            CubeCountStrategy::Flat {
                m_cubes,
                n_cubes,
                batch_cubes: _,
            } => self.absolute_index_to_m_n_batch(
                CUBE_POS_X,
                *m_cubes,
                *n_cubes,
                global_partitioning,
            ),
        }
    }

    fn absolute_index_to_m_n_batch(
        &self,
        absolute_index: u32,
        m_cubes: u32,
        n_cubes: u32,
        #[comptime] global_partitioning: GlobalPartitioning,
    ) -> (u32, u32, u32) {
        let batch_stride = m_cubes * n_cubes;
        let batch_pos = absolute_index / batch_stride;
        let matrix_pos = absolute_index % batch_stride;

        let (m_pos, n_pos) = match comptime!(global_partitioning) {
            GlobalPartitioning::Natural => (matrix_pos / n_cubes, matrix_pos % n_cubes),
            GlobalPartitioning::Transposed => (matrix_pos % m_cubes, matrix_pos / m_cubes),
        };

        (m_pos, n_pos, batch_pos)
    }
}
