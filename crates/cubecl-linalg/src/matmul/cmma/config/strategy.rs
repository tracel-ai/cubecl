#[derive(Clone, Copy)]
/// Defines how data travels from accumulators to global output
pub enum WriteOutStrategy {
    /// Accumulators for one warp are put concurrently in a shared memory large enough to contain them all
    LargeSmem,
    /// Accumulators for one warp are put sequentially in a shared memory with only one reusable spot
    ReuseSmem,
}

impl From<WriteOutStrategy> for u32 {
    fn from(value: WriteOutStrategy) -> Self {
        match value {
            WriteOutStrategy::LargeSmem => 0,
            WriteOutStrategy::ReuseSmem => 1,
        }
    }
}

/// How cubes are dispatched in the hypercube
/// Should impact L2 cache reuse
#[derive(Clone, Copy)]
pub enum CubeDispatchStrategy {
    /// Cubes are dispatched row major
    RowMajor,
    /// Cubes are dispatched col major
    ColMajor,
    /// Cubes follow swizzle pattern, see https://bruce-lee-ly.medium.com/nvidia-tensor-core-cuda-hgemm-advanced-optimization-5a17eb77dd85
    Swizzle,
}
impl CubeDispatchStrategy {
    pub(crate) fn get_cube_dim(&self, num_rows: usize, num_cols: usize, b_mn: usize) -> (u32, u32) {
        match self {
            CubeDispatchStrategy::RowMajor | CubeDispatchStrategy::Swizzle => (
                f32::ceil(num_cols as f32 / b_mn as f32) as u32,
                f32::ceil(num_rows as f32 / b_mn as f32) as u32,
            ),
            CubeDispatchStrategy::ColMajor => (
                f32::ceil(num_rows as f32 / b_mn as f32) as u32,
                f32::ceil(num_cols as f32 / b_mn as f32) as u32,
            ),
        }
    }
}

impl From<CubeDispatchStrategy> for u32 {
    fn from(value: CubeDispatchStrategy) -> Self {
        match value {
            CubeDispatchStrategy::RowMajor => 0,
            CubeDispatchStrategy::ColMajor => 1,
            CubeDispatchStrategy::Swizzle => 2,
        }
    }
}

#[derive(Clone, Copy)]
/// Defines how data travels from accumulators to global output
pub enum ComputeLoopOrderStrategy {
    /// Accumulators for one warp are put concurrently in a shared memory large enough to contain them all
    AllBuffersFirst,
    /// Accumulators for one warp are put sequentially in a shared memory with only one reusable spot
    AllAccumulatorsFirst(bool),
}

impl From<ComputeLoopOrderStrategy> for (u32, bool) {
    fn from(value: ComputeLoopOrderStrategy) -> Self {
        match value {
            ComputeLoopOrderStrategy::AllBuffersFirst => (0, false),
            ComputeLoopOrderStrategy::AllAccumulatorsFirst(reuse_lhs_fragment) => {
                (1, reuse_lhs_fragment)
            }
        }
    }
}

#[derive(Clone, Copy)]
#[allow(clippy::enum_variant_names)]
/// Defines how data is loaded from global to shared memory
pub enum SmemLoaderStrategy {
    /// One coop fills one tile, tile order is row major
    TilewiseRowMajor,
    /// One coop fills one tile, tile order is col major
    TilewiseColMajor,
    /// Coops can work in any tile, tile order is row major
    ContinuousRowMajor,
    /// Coops can work in any tile, tile order is col major
    ContinuousColMajor,
}

impl From<SmemLoaderStrategy> for u32 {
    fn from(value: SmemLoaderStrategy) -> Self {
        match value {
            SmemLoaderStrategy::TilewiseRowMajor => 0,
            SmemLoaderStrategy::TilewiseColMajor => 1,
            SmemLoaderStrategy::ContinuousRowMajor => 2,
            SmemLoaderStrategy::ContinuousColMajor => 3,
        }
    }
}

#[derive(Clone, Copy)]
/// Defines if different coops have different roles
pub enum BlockLoopStrategy {
    /// All coops both load and compute
    Standard(u32),
    /// Part compute, part load
    Split(u32, u32),
}

impl From<BlockLoopStrategy> for (u32, u32, u32) {
    fn from(value: BlockLoopStrategy) -> Self {
        match value {
            BlockLoopStrategy::Standard(num_coops) => (0, num_coops, num_coops),
            BlockLoopStrategy::Split(num_compute, num_load) => (1, num_compute, num_load),
        }
    }
}

impl BlockLoopStrategy {
    pub(crate) fn num_coops(&self) -> u32 {
        match self {
            BlockLoopStrategy::Standard(num_coops) => *num_coops,
            BlockLoopStrategy::Split(num_compute, num_load) => num_compute + num_load,
        }
    }
}
