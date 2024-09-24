#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
/// Defines how data travels from accumulators to global output
pub enum WriteOutStrategy {
    /// Accumulators for one warp are put concurrently in a shared memory large enough to contain them all
    LargeSmem,
    /// Accumulators for one warp are put sequentially in a shared memory with only one reusable spot
    ReuseSmem,
}

/// How cubes are dispatched in the hypercube
/// Should impact L2 cache reuse
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub enum RasterizationStrategy {
    /// Cubes are dispatched row major
    RowMajor,
    /// Cubes are dispatched col major
    ColMajor,
    /// Cubes follow swizzle pattern
    Swizzle,
}

impl RasterizationStrategy {
    pub(crate) fn get_cube_dim(
        &self,
        num_rows: usize,
        num_cols: usize,
        b_m: usize,
        b_n: usize,
    ) -> (u32, u32) {
        let cubes_for_rows = f32::ceil(num_rows as f32 / b_m as f32) as u32;
        let cubes_for_cols = f32::ceil(num_cols as f32 / b_n as f32) as u32;

        match self {
            RasterizationStrategy::RowMajor | RasterizationStrategy::Swizzle => {
                (cubes_for_cols, cubes_for_rows)
            }
            RasterizationStrategy::ColMajor => (cubes_for_rows, cubes_for_cols),
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
/// Defines how data travels from accumulators to global output
pub enum ComputeLoopOrderStrategy {
    /// Accumulators for one warp are put concurrently in a shared memory large enough to contain them all
    AllBuffersFirst,
    /// Accumulators for one warp are put sequentially in a shared memory with only one reusable spot
    AllAccumulatorsFirst(bool),
}

// impl From<ComputeLoopOrderStrategy> for (u32, bool) {
//     fn from(value: ComputeLoopOrderStrategy) -> Self {
//         match value {
//             ComputeLoopOrderStrategy::AllBuffersFirst => (0, false),
//             ComputeLoopOrderStrategy::AllAccumulatorsFirst(reuse_lhs_fragment) => {
//                 (1, reuse_lhs_fragment)
//             }
//         }
//     }
// }

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub enum TilingOrderStrategy {
    RowMajor,
    ColMajor,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
/// Defines how data is loaded from global to shared memory
pub enum SmemLoaderStrategy {
    /// One coop fills one tile
    Tilewise(TilingOrderStrategy),
    /// Coops can work in any tile
    Continuous(TilingOrderStrategy),
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
/// Defines if different coops have different roles
pub enum MainLoopStrategy {
    /// All coops both load and compute
    Standard(u32),
    /// Part compute, part load
    Split(u32, u32),
}

impl MainLoopStrategy {
    pub(crate) fn get_num_planes(&self) -> (u32, u32) {
        match self {
            MainLoopStrategy::Standard(num_coops) => (*num_coops, *num_coops),
            MainLoopStrategy::Split(num_compute, num_load) => (*num_compute, *num_load),
        }
    }
}
