use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_std::div_ceil;

use crate::components::batch::swizzle;
use crate::components::{MatmulProblem, TilingScheme};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum GlobalPartitioning {
    Natural,
    Transposed,
    // input is swizzle width
    SwizzleNatural(u32),
    // input is swizzle width
    SwizzleTranspose(u32),
}

/// Maps the needed cubes in m/n/batch to num_sms/cubes_per_sm
#[derive(Clone)]
pub struct CubeCounterConfig {
    num_sms: u32,
    cube_span: CubeSpan,
    global_partitioning: GlobalPartitioning,
}

#[derive(Clone)]
pub struct CubeSpan {
    m: u32,
    n: u32,
    batch: u32,
}

impl CubeCounterConfig {
    pub fn new(
        num_sms: u32,
        tiling_scheme: &TilingScheme,
        global_partitioning: GlobalPartitioning,
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
        }
    }

    pub fn cube_count(&self, problem: &MatmulProblem) -> CubeCount {
        let m_cubes = (problem.m as u32).div_ceil(self.cube_span.m);
        let n_cubes = (problem.n as u32).div_ceil(self.cube_span.n);
        let batch_cubes = (problem.num_batches() as u32).div_ceil(self.cube_span.batch);

        let total_cubes = m_cubes * n_cubes * batch_cubes;
        let num_sms_used = gcd(total_cubes, self.num_sms);
        let cubes_per_sm = total_cubes / num_sms_used;

        CubeCount::Static(num_sms_used, cubes_per_sm, 1)
    }
}

#[derive(CubeType, Clone)]
pub struct CubeCounter {
    m_cubes: u32,
    n_cubes: u32,
    batch_cubes: u32,
    cubes_per_sm: u32,
    #[cube(comptime)]
    global_partitioning: GlobalPartitioning,
}

#[cube]
impl CubeCounter {
    pub fn new(
        m_shape: u32,
        n_shape: u32,
        batch_shape: u32,
        #[comptime] cube_counter_config: CubeCounterConfig,
    ) -> CubeCounter {
        let m_cubes = div_ceil(m_shape, cube_counter_config.cube_span.m);
        let n_cubes = div_ceil(n_shape, cube_counter_config.cube_span.n);
        let batch_cubes = div_ceil(batch_shape, cube_counter_config.cube_span.batch);

        // TODO take something in scalar input to not compute this
        let cubes_per_sm =
            gcd_cube_tmp(m_cubes * n_cubes * batch_cubes, cube_counter_config.num_sms);

        CubeCounter {
            m_cubes,
            n_cubes,
            batch_cubes,
            cubes_per_sm,
            global_partitioning: cube_counter_config.global_partitioning,
        }
    }

    /// Given a cube position (SM ID, local index), returns the tensor coordinates (m, n, batch).
    pub fn cube_pos_to_tensor_pos(&self) -> (u32, u32, u32) {
        let sm_id = CUBE_POS_X;
        let index_within_sm = CUBE_POS_Y;
        let absolute_index = sm_id * self.cubes_per_sm + index_within_sm;

        let batch_stride = self.n_cubes * self.m_cubes;
        let batch_pos = absolute_index / batch_stride;
        let rem = absolute_index % batch_stride;

        let (m_pos, n_pos) = match comptime!(self.global_partitioning) {
            GlobalPartitioning::Natural => {
                let m_stride = self.n_cubes;
                (rem / m_stride, rem % m_stride)
            }
            GlobalPartitioning::Transposed => {
                let n_stride = self.m_cubes;
                (rem / n_stride, rem % n_stride)
            }
            GlobalPartitioning::SwizzleNatural(swizzle_width) => {
                swizzle(rem, self.n_cubes, swizzle_width)
            }
            GlobalPartitioning::SwizzleTranspose(swizzle_width) => {
                let (n_pos, m_pos) = swizzle(rem, self.m_cubes, swizzle_width);
                (m_pos, n_pos)
            }
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

#[cube]
fn gcd_cube_tmp(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}
