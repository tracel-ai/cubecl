mod partition;
mod partitioned_matmul;
mod plane_partitioned;
mod scheduler;
mod unit_partitioned;

pub use partitioned_matmul::StagePartitioner;
pub use plane_partitioned::{PlaneMatmulFamily, PlanePartitioner};
pub use scheduler::{PartitionScheduler, PartitionSchedulerScheme};
pub use unit_partitioned::{UnitMatmulFamily, UnitPartitioner};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Number of stages in one shared memory, i.e. buffers for double buffering
pub struct NumStages {
    pub lhs: u32,
    pub rhs: u32,
}

impl From<(u32, u32)> for NumStages {
    fn from(value: (u32, u32)) -> Self {
        NumStages {
            lhs: value.0,
            rhs: value.1,
        }
    }
}
