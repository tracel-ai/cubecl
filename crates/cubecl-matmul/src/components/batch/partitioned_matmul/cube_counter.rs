use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

use crate::components::{MatmulProblem, TilingScheme};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
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
    SmPerCubeFirst(u32),

    /// X: num cubes per SM, Y: num SMs
    /// Given value: num SMs
    CubePerSmFirst(u32),

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
    },
    CubePerSmFirst {
        num_sms_used: u32,
        cubes_per_sm: u32,
        m_cubes: u32,
        n_cubes: u32,
    },
    Flat {
        total_cubes: u32,
        m_cubes: u32,
        n_cubes: u32,
    },
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
        let total_cubes = m_cubes * n_cubes * batch_cubes;

        match self.cube_pos_strategy {
            CubeCountStrategyConfig::FromProblem => CubeCountStrategy::FromProblem {
                m_cubes,
                n_cubes,
                batch_cubes,
            },
            CubeCountStrategyConfig::SmPerCubeFirst(num_sms) => {
                let num_sms_used = gcd(total_cubes, num_sms);
                let cubes_per_sm = total_cubes / num_sms_used;
                CubeCountStrategy::SmPerCubeFirst {
                    num_sms_used,
                    cubes_per_sm,
                    m_cubes,
                    n_cubes,
                }
            }
            CubeCountStrategyConfig::CubePerSmFirst(num_sms) => {
                let num_sms_used = gcd(total_cubes, num_sms);
                let cubes_per_sm = total_cubes / num_sms_used;
                CubeCountStrategy::CubePerSmFirst {
                    num_sms_used,
                    cubes_per_sm,
                    m_cubes,
                    n_cubes,
                }
            }
            CubeCountStrategyConfig::Flat => CubeCountStrategy::Flat {
                total_cubes,
                m_cubes,
                n_cubes,
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
            } => CubeCount::Static(*num_sms_used, *cubes_per_sm, 1),
            CubeCountStrategy::CubePerSmFirst {
                num_sms_used,
                cubes_per_sm,
                m_cubes: _,
                n_cubes: _,
            } => CubeCount::Static(*cubes_per_sm, *num_sms_used, 1),
            CubeCountStrategy::Flat {
                total_cubes,
                m_cubes: _,
                n_cubes: _,
            } => CubeCount::Static(*total_cubes, 1, 1),
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
            } => CubeCountStrategyArgs::SmPerCubeFirst {
                num_sms_used: ScalarArg::new(*num_sms_used),
                cubes_per_sm: ScalarArg::new(*cubes_per_sm),
                m_cubes: ScalarArg::new(*m_cubes),
                n_cubes: ScalarArg::new(*n_cubes),
            },
            CubeCountStrategy::CubePerSmFirst {
                num_sms_used,
                cubes_per_sm,
                m_cubes,
                n_cubes,
            } => CubeCountStrategyArgs::CubePerSmFirst {
                num_sms_used: ScalarArg::new(*num_sms_used),
                cubes_per_sm: ScalarArg::new(*cubes_per_sm),
                m_cubes: ScalarArg::new(*m_cubes),
                n_cubes: ScalarArg::new(*n_cubes),
            },
            CubeCountStrategy::Flat {
                total_cubes,
                m_cubes,
                n_cubes,
            } => CubeCountStrategyArgs::Flat {
                total_cubes: ScalarArg::new(*total_cubes),
                m_cubes: ScalarArg::new(*m_cubes),
                n_cubes: ScalarArg::new(*n_cubes),
            },
        }
    }
}

#[cube]
impl CubeCountStrategy {
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
                cubes_per_sm,
                m_cubes,
                n_cubes,
            } => self.absolute_index_to_m_n_batch(
                CUBE_POS_X * cubes_per_sm + CUBE_POS_Y,
                *m_cubes,
                *n_cubes,
                global_partitioning,
            ),
            CubeCountStrategy::CubePerSmFirst {
                num_sms_used: _,
                cubes_per_sm,
                m_cubes,
                n_cubes,
            } => self.absolute_index_to_m_n_batch(
                CUBE_POS_Y * cubes_per_sm + CUBE_POS_X,
                *m_cubes,
                *n_cubes,
                global_partitioning,
            ),
            CubeCountStrategy::Flat {
                total_cubes: _,
                m_cubes,
                n_cubes,
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

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}
