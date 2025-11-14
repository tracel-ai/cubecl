use crate::components::PartitionSize;

use cubecl::prelude::*;
use cubecl_core as cubecl;

/// Defines how partition indices are scheduled across axes.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum PartitionSchedulerScheme {
    /// Rotates indices per axis to stagger partitions and reduce shared memory conflicts.
    Offset,
    /// Maps partitions in simple row-major order without rotations.
    Naive,
}

/// Schedules global indices for M, N, and K axes in a partitioned matmul.
/// Internally uses an `AxisScheduler` per axis.
#[derive(CubeType)]
pub struct PartitionScheduler {
    pub m: AxisScheduler,
    pub n: AxisScheduler,
    pub k: AxisScheduler,
}

#[cube]
impl PartitionScheduler {
    /// Creates a `PartitionScheduler` for a partition at (partition_index_m, partition_index_n).
    ///
    /// - `Offset`: rotates indices per axis to spread memory access and avoid conflicts.
    /// - `Naive`: uses simple row-major order for partitions.
    pub fn new(
        partition_index_m: u32,
        partition_index_n: u32,
        #[comptime] partition_size: PartitionSize,
        #[comptime] partition_schedule_scheme: PartitionSchedulerScheme,
    ) -> PartitionScheduler {
        match partition_schedule_scheme {
            PartitionSchedulerScheme::Offset => {
                // M-axis rotation: ensures partitions in the same row start at different M tiles.
                let m_offset = (partition_index_n / partition_size.k()) % partition_size.m();

                // N-axis rotation: ensures partitions in the same column start at different N tiles.
                let n_offset = (partition_index_m / partition_size.k()) % partition_size.n();

                // K-axis rotation: simple offset; same diagonal can share K safely.
                let k_offset = (partition_index_m + partition_index_n) % partition_size.k();

                PartitionScheduler {
                    m: AxisScheduler::new_Offset(OffsetAxisScheduler::new(
                        m_offset,
                        partition_index_m,
                        partition_size.m(),
                    )),
                    n: AxisScheduler::new_Offset(OffsetAxisScheduler::new(
                        n_offset,
                        partition_index_n,
                        partition_size.n(),
                    )),
                    k: AxisScheduler::new_Offset(OffsetAxisScheduler::new(
                        k_offset,
                        0u32,
                        partition_size.k(),
                    )),
                }
            }
            PartitionSchedulerScheme::Naive => PartitionScheduler {
                m: AxisScheduler::new_Naive(NaiveAxisScheduler::new(
                    partition_index_m,
                    partition_size.m(),
                )),
                n: AxisScheduler::new_Naive(NaiveAxisScheduler::new(
                    partition_index_n,
                    partition_size.n(),
                )),
                k: AxisScheduler::new_Naive(NaiveAxisScheduler::new(0u32, partition_size.k())),
            },
        }
    }

    /// Maps a local M index to a global index.
    pub fn map_m(&self, i: u32) -> u32 {
        self.m.map(i)
    }

    /// Maps a local N index to a global index.
    pub fn map_n(&self, i: u32) -> u32 {
        self.n.map(i)
    }

    /// Maps a local K index to a global index.
    pub fn map_k(&self, i: u32) -> u32 {
        self.k.map(i)
    }
}

/// Axis-specific scheduler that delegates to either `OffsetAxisScheduler` or `NaiveAxisScheduler`.
#[derive(CubeType)]
#[allow(unused)]
pub enum AxisScheduler {
    Offset(OffsetAxisScheduler),
    Naive(NaiveAxisScheduler),
}

/// Schedules indices for one axis with rotation and wrapping.
///
/// Combines:
/// - `inner_offset`: rotation inside this partition.
/// - `outer_offset`: global shift for skipping previous partitions.
#[derive(CubeType)]
pub struct OffsetAxisScheduler {
    inner_offset: u32,
    outer_offset: u32,
    #[cube(comptime)]
    len: u32,
}

/// Schedules indices for one axis in row-major order.
/// Just adds a global shift based on the partition index.
#[derive(CubeType)]
pub struct NaiveAxisScheduler {
    outer_offset: u32,
}

#[cube]
impl AxisScheduler {
    pub fn map(&self, i: u32) -> u32 {
        match self {
            AxisScheduler::Offset(offset_axis_scheduler) => offset_axis_scheduler.map(i),
            AxisScheduler::Naive(naive_axis_scheduler) => naive_axis_scheduler.map(i),
        }
    }
}

#[cube]
impl OffsetAxisScheduler {
    pub fn new(
        inner_offset: u32,
        partition_index: u32,
        #[comptime] len: u32,
    ) -> OffsetAxisScheduler {
        let outer_offset = partition_index * len;
        OffsetAxisScheduler {
            inner_offset,
            outer_offset,
            len,
        }
    }

    pub fn map(&self, i: u32) -> u32 {
        let relative = (i + self.inner_offset) % self.len;
        relative + self.outer_offset
    }
}

#[cube]
impl NaiveAxisScheduler {
    pub fn new(partition_index: u32, #[comptime] len: u32) -> NaiveAxisScheduler {
        let outer_offset = partition_index * len;
        NaiveAxisScheduler { outer_offset }
    }

    pub fn map(&self, i: u32) -> u32 {
        i + self.outer_offset
    }
}
