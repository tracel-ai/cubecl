use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_std::div_ceil;

use crate::components::{MatmulProblem, TilingScheme};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum GlobalPartitioning {
    Natural,
    Transposed,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum CubePosStrategy {
    /// X: num cubes in m, Y: num cubes in n, Z: num cubes in batch
    FromProblem,

    /// X: num SMs, Y: num cubes per SM
    SmPerCubeFirst,

    /// X: num cubes per SM, Y: num SMs
    CubePerSmFirst,

    /// X: total cubes flattened (num SMs * num cubes per SM)
    Flat,
}

/// Maps the needed cubes in m/n/batch to num_sms/cubes_per_sm
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct CubeCounterConfig {
    num_sms: u32,
    cube_span: CubeSpan,
    global_partitioning: GlobalPartitioning,
    cube_pos_strategy: CubePosStrategy,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct CubeSpan {
    m: u32,
    n: u32,
    batch: u32,
}

#[derive(Debug)]
pub struct CubeCountData {
    m_cubes: u32,
    n_cubes: u32,
    batch_cubes: u32,
    total_cubes: u32,
    num_sms_used: u32,
    cubes_per_sm: u32,
    cube_pos_strategy: CubePosStrategy,
}

#[derive(CubeType, Clone)]
pub struct CubeCounter {
    m_cubes: u32,
    n_cubes: u32,
    batch_cubes: u32,
    cubes_per_sm: u32,
    #[cube(comptime)]
    global_partitioning: GlobalPartitioning,
    #[cube(comptime)]
    cube_pos_strategy: CubePosStrategy,
}

#[derive(CubeType, CubeLaunch)]
pub struct CubeCountArgs {
    num_sms_used: u32,
    cubes_per_sm: u32,
    max_activated_cube_count: u32,
}

impl CubeCounterConfig {
    pub fn new(
        num_sms: u32,
        tiling_scheme: &TilingScheme,
        global_partitioning: GlobalPartitioning,
        cube_pos_strategy: CubePosStrategy,
    ) -> Self {
        let cube_span = CubeSpan {
            m: tiling_scheme.elements_in_global_partition_m(),
            n: tiling_scheme.elements_in_global_partition_n(),
            batch: tiling_scheme.global_partition_size.batches,
        };

        Self {
            num_sms,
            cube_span,
            global_partitioning,
            cube_pos_strategy,
        }
    }

    pub fn cube_count_data(&self, problem: &MatmulProblem) -> CubeCountData {
        let m_cubes = (problem.m as u32).div_ceil(self.cube_span.m);
        let n_cubes = (problem.n as u32).div_ceil(self.cube_span.n);
        let batch_cubes = (problem.num_batches() as u32).div_ceil(self.cube_span.batch);

        let total_cubes = m_cubes * n_cubes * batch_cubes;
        let num_sms_used = gcd(total_cubes, self.num_sms);
        let cubes_per_sm = total_cubes / num_sms_used;

        CubeCountData {
            m_cubes,
            n_cubes,
            batch_cubes,
            total_cubes,
            num_sms_used,
            cubes_per_sm,
            cube_pos_strategy: self.cube_pos_strategy,
        }
    }
}

impl CubeCountData {
    pub fn to_cube_count(&self) -> CubeCount {
        match self.cube_pos_strategy {
            CubePosStrategy::FromProblem => {
                CubeCount::Static(self.m_cubes, self.n_cubes, self.batch_cubes)
            }
            CubePosStrategy::SmPerCubeFirst => {
                CubeCount::Static(self.num_sms_used, self.cubes_per_sm, 1)
            }
            CubePosStrategy::CubePerSmFirst => {
                CubeCount::Static(self.cubes_per_sm, self.num_sms_used, 1)
            }
            CubePosStrategy::Flat => CubeCount::Static(self.total_cubes, 1, 1),
        }
    }

    pub fn to_args<'a, R: Runtime>(&self) -> CubeCountArgsLaunch<'a, R> {
        CubeCountArgsLaunch::new(
            ScalarArg::new(self.num_sms_used),
            ScalarArg::new(self.cubes_per_sm),
            // TODO
            ScalarArg::new(0),
        )
    }
}

#[cube]
impl CubeCounter {
    pub fn new(
        m_shape: u32,
        n_shape: u32,
        batch_shape: u32,
        cube_count_args: CubeCountArgs,
        #[comptime] cube_counter_config: CubeCounterConfig,
    ) -> CubeCounter {
        let m_cubes = div_ceil(m_shape, cube_counter_config.cube_span.m);
        let n_cubes = div_ceil(n_shape, cube_counter_config.cube_span.n);
        let batch_cubes = div_ceil(batch_shape, cube_counter_config.cube_span.batch);
        let total_cubes = m_cubes * n_cubes * batch_cubes;
        let cubes_per_sm = total_cubes / cube_count_args.num_sms_used;

        CubeCounter {
            m_cubes,
            n_cubes,
            batch_cubes,
            cubes_per_sm,
            global_partitioning: cube_counter_config.global_partitioning,
            cube_pos_strategy: cube_counter_config.cube_pos_strategy,
        }
    }

    /// Given a cube position (SM ID, local index), returns the tensor coordinates (m, n, batch).
    pub fn cube_pos_to_tensor_pos(&self) -> (u32, u32, u32) {
        match comptime!(self.cube_pos_strategy) {
            CubePosStrategy::FromProblem => (CUBE_POS_X, CUBE_POS_Y, CUBE_POS_Z),
            _ => {
                let index = match comptime!(self.cube_pos_strategy) {
                    CubePosStrategy::SmPerCubeFirst => CUBE_POS_X * self.cubes_per_sm + CUBE_POS_Y,
                    CubePosStrategy::CubePerSmFirst => CUBE_POS_Y * self.cubes_per_sm + CUBE_POS_X,
                    CubePosStrategy::Flat => CUBE_POS_X,
                    _ => comptime!(unreachable!()),
                };

                let batch_stride = self.n_cubes * self.m_cubes;
                let batch_pos = index / batch_stride;
                let matrix_pos = index % batch_stride;

                let (m_pos, n_pos) = match comptime!(self.global_partitioning) {
                    GlobalPartitioning::Natural => {
                        let m_stride = self.n_cubes;
                        (matrix_pos / m_stride, matrix_pos % m_stride)
                    }
                    GlobalPartitioning::Transposed => {
                        let n_stride = self.m_cubes;
                        (matrix_pos % n_stride, matrix_pos / n_stride)
                    }
                };

                (m_pos, n_pos, batch_pos)
            }
        }
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
