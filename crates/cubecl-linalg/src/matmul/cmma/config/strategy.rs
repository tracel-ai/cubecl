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
        b_m: u32,
        b_n: u32,
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
    /// One plane fills one tile
    Tilewise(TilingOrderStrategy),
    /// Planes can work in any tile
    Continuous(TilingOrderStrategy),
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
/// Defines if different planes have different roles
pub enum MainLoopStrategy {
    /// All planes both load and compute
    Standard,
    /// Part compute, part load. Number of load planes specified here
    Split(u32),
}

impl MainLoopStrategy {
    pub(crate) fn get_num_load_planes(&self, num_compute_planes: u32) -> u32 {
        match self {
            MainLoopStrategy::Standard => num_compute_planes,
            MainLoopStrategy::Split(num_load_planes) => *num_load_planes,
        }
    }

    pub(crate) fn get_num_planes(&self, num_compute_planes: u32) -> u32 {
        num_compute_planes
            + match self {
                MainLoopStrategy::Standard => 0,
                MainLoopStrategy::Split(num_load) => *num_load,
            }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
/// Defines the dimension of tiles consumed by tensor cores
pub enum TileDimensionStrategy {
    // M: 16, K: 16, N: 16
    M16K16N16,
    // M: 32, K: 16, N: 8
    // Doesn't work
    M32K16N8,
    // M: 8, K: 16, N: 32
    // Doesn't work
    M8K16N32,
}

pub struct TileDimension {
    pub m: u32,
    pub k: u32,
    pub n: u32,
}

impl From<TileDimensionStrategy> for TileDimension {
    fn from(value: TileDimensionStrategy) -> Self {
        match value {
            TileDimensionStrategy::M16K16N16 => Self {
                m: 16,
                k: 16,
                n: 16,
            },
            TileDimensionStrategy::M32K16N8 => panic!("Unsupported, contains a bug"),
            TileDimensionStrategy::M8K16N32 => panic!("Unsupported, contains a bug"),
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
/// Defines how many compute planes there should be
pub enum NumComputePlanesStrategy {
    /// As many as tiles in LHS smem
    NumTilesLhs,
    /// As many as rows of tiles in LHS smem
    NumTilesM,
}

impl NumComputePlanesStrategy {
    pub(crate) fn get_num_compute_planes(&self, num_tiles_m: u32, num_tiles_k: u32) -> u32 {
        match self {
            NumComputePlanesStrategy::NumTilesLhs => num_tiles_m * num_tiles_k,
            NumComputePlanesStrategy::NumTilesM => num_tiles_m,
        }
    }

    pub(crate) fn get_num_accumulators(
        &self,
        num_tiles_m: u32,
        num_tiles_k: u32,
        num_tiles_n: u32,
    ) -> u32 {
        num_tiles_m * num_tiles_n / self.get_num_compute_planes(num_tiles_m, num_tiles_k)
    }
}
