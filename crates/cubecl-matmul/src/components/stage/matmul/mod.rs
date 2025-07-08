mod partition;
mod partitioned_matmul;
mod plane_partitioned;
mod unit_partitioned;

pub use plane_partitioned::PlaneMatmulFamily;
pub use unit_partitioned::UnitMatmulFamily;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Number of stages in one shared memory, i.e. buffers for double buffering
pub struct NumStages {
    lhs: u32,
    rhs: u32,
}

impl From<(u32, u32)> for NumStages {
    fn from(value: (u32, u32)) -> Self {
        NumStages {
            lhs: value.0,
            rhs: value.1,
        }
    }
}
