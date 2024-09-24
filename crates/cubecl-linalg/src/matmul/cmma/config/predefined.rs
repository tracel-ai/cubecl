use super::{
    strategy::{
        BlockLoopStrategy, ComputeLoopOrderStrategy, CubeDispatchStrategy, SmemLoaderStrategy,
        WriteOutStrategy,
    },
    CmmaConfig,
};

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub enum PredefinedCmmaConfig {
    M128K16,
    M128K16N64,
    M64K32,
    M64K32N32,
    M64K16,
    M64K16N32,
    M32K16,
    M32K16N64,
    M32K32,
    M16K32N64,
    SplitM32k32,
    SplitM64k16,
    TilewiseInverted,
    Continuous,
    ContinuousInverted,
    LargeSmem,
    RowMajorDispatch,
    SwizzleDispatch,
    AccumulatorsFirstNoReuse,
    AccumulatorsFirstWithReuse,
}

impl From<PredefinedCmmaConfig> for CmmaConfig {
    fn from(val: PredefinedCmmaConfig) -> Self {
        match val {
            // Probably the fastest
            PredefinedCmmaConfig::M128K16 => CmmaConfig {
                b_m: 128,
                b_k: 16,
                b_n: 128,
                block_loop_strategy: BlockLoopStrategy::Standard(8),
                ..Default::default()
            },
            PredefinedCmmaConfig::M64K32 => CmmaConfig {
                b_m: 64,
                b_k: 32,
                b_n: 64,
                block_loop_strategy: BlockLoopStrategy::Standard(8),
                ..Default::default()
            },

            PredefinedCmmaConfig::M64K16 => CmmaConfig {
                b_m: 64,
                b_k: 16,
                b_n: 64,
                block_loop_strategy: BlockLoopStrategy::Standard(4),
                ..Default::default()
            },
            PredefinedCmmaConfig::M32K16 => CmmaConfig {
                b_m: 32,
                b_k: 16,
                b_n: 32,
                block_loop_strategy: BlockLoopStrategy::Standard(2),
                ..Default::default()
            },
            PredefinedCmmaConfig::M32K32 => CmmaConfig {
                b_m: 32,
                b_k: 32,
                b_n: 32,
                block_loop_strategy: BlockLoopStrategy::Standard(4),
                ..Default::default()
            },
            PredefinedCmmaConfig::SplitM32k32 => CmmaConfig {
                b_m: 32,
                b_k: 32,
                b_n: 32,
                block_loop_strategy: BlockLoopStrategy::Split(4, 4),
                ..Default::default()
            },
            PredefinedCmmaConfig::SplitM64k16 => CmmaConfig {
                b_m: 64,
                b_k: 16,
                b_n: 64,
                block_loop_strategy: BlockLoopStrategy::Split(4, 4),
                ..Default::default()
            },
            PredefinedCmmaConfig::TilewiseInverted => CmmaConfig {
                lhs_smem_loader_strategy: SmemLoaderStrategy::TilewiseColMajor,
                rhs_smem_loader_strategy: SmemLoaderStrategy::TilewiseRowMajor,
                ..Default::default()
            },
            PredefinedCmmaConfig::Continuous => CmmaConfig {
                lhs_smem_loader_strategy: SmemLoaderStrategy::ContinuousRowMajor,
                rhs_smem_loader_strategy: SmemLoaderStrategy::ContinuousColMajor,
                ..Default::default()
            },
            PredefinedCmmaConfig::ContinuousInverted => CmmaConfig {
                lhs_smem_loader_strategy: SmemLoaderStrategy::ContinuousColMajor,
                rhs_smem_loader_strategy: SmemLoaderStrategy::ContinuousRowMajor,
                ..Default::default()
            },
            PredefinedCmmaConfig::LargeSmem => CmmaConfig {
                write_out_strategy: WriteOutStrategy::LargeSmem,
                ..Default::default()
            },
            PredefinedCmmaConfig::RowMajorDispatch => CmmaConfig {
                cube_dispatch_strategy: CubeDispatchStrategy::RowMajor,
                ..Default::default()
            },
            PredefinedCmmaConfig::SwizzleDispatch => CmmaConfig {
                cube_dispatch_strategy: CubeDispatchStrategy::Swizzle,
                ..Default::default()
            },
            PredefinedCmmaConfig::AccumulatorsFirstNoReuse => CmmaConfig {
                compute_loop_order_strategy: ComputeLoopOrderStrategy::AllAccumulatorsFirst(false),
                ..Default::default()
            },
            PredefinedCmmaConfig::AccumulatorsFirstWithReuse => CmmaConfig {
                compute_loop_order_strategy: ComputeLoopOrderStrategy::AllAccumulatorsFirst(true),
                ..Default::default()
            },
            PredefinedCmmaConfig::M128K16N64 => CmmaConfig {
                b_m: 128,
                b_k: 16,
                b_n: 64,
                block_loop_strategy: BlockLoopStrategy::Standard(8),
                ..Default::default()
            },
            PredefinedCmmaConfig::M64K32N32 => CmmaConfig {
                b_m: 64,
                b_k: 32,
                b_n: 32,
                block_loop_strategy: BlockLoopStrategy::Standard(8),
                ..Default::default()
            },
            PredefinedCmmaConfig::M64K16N32 => CmmaConfig {
                b_m: 64,
                b_k: 16,
                b_n: 32,
                block_loop_strategy: BlockLoopStrategy::Standard(4),
                ..Default::default()
            },
            PredefinedCmmaConfig::M32K16N64 => CmmaConfig {
                b_m: 32,
                b_k: 16,
                b_n: 64,
                rhs_smem_loader_strategy: SmemLoaderStrategy::ContinuousColMajor,
                block_loop_strategy: BlockLoopStrategy::Standard(2),
                ..Default::default()
            },
            PredefinedCmmaConfig::M16K32N64 => CmmaConfig {
                b_m: 16,
                b_k: 32,
                b_n: 64,
                rhs_smem_loader_strategy: SmemLoaderStrategy::ContinuousColMajor,
                block_loop_strategy: BlockLoopStrategy::Standard(2),
                ..Default::default()
            },
        }
    }
}
