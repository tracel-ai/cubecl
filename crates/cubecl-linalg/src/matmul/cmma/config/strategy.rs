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
    Standard,
    /// Half coops load, half compute
    Split,
}

impl From<BlockLoopStrategy> for u32 {
    fn from(value: BlockLoopStrategy) -> Self {
        match value {
            BlockLoopStrategy::Standard => 0,
            BlockLoopStrategy::Split => 1,
        }
    }
}
