use super::{
    strategy::{
        ComputeLoopOrderStrategy, MainLoopStrategy, RasterizationStrategy, SmemLoaderStrategy,
        WriteOutStrategy,
    },
    CmmaConfig, TilingOrderStrategy,
};

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub enum PredefinedCmmaConfig {
    M128K16,
    M64K32,
    M64K16,
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
    RowMajorRasterization,
    SwizzleRasterization,
    AccumulatorsFirstNoReuse,
    BuffersFirst,
}

impl From<PredefinedCmmaConfig> for CmmaConfig {
    fn from(val: PredefinedCmmaConfig) -> Self {
        match val {
            // Probably the fastest
            PredefinedCmmaConfig::M128K16 => CmmaConfig {
                b_m: 128,
                b_k: 16,
                b_n: 128,
                ..Default::default()
            },
            PredefinedCmmaConfig::M64K32 => CmmaConfig {
                b_m: 64,
                b_k: 32,
                b_n: 64,
                ..Default::default()
            },

            PredefinedCmmaConfig::M64K16 => CmmaConfig {
                b_m: 64,
                b_k: 16,
                b_n: 64,
                ..Default::default()
            },
            PredefinedCmmaConfig::M32K16 => CmmaConfig {
                b_m: 32,
                b_k: 16,
                b_n: 32,
                ..Default::default()
            },
            PredefinedCmmaConfig::M32K32 => CmmaConfig {
                b_m: 32,
                b_k: 32,
                b_n: 32,
                ..Default::default()
            },
            PredefinedCmmaConfig::SplitM32k32 => CmmaConfig {
                b_m: 32,
                b_k: 32,
                b_n: 32,
                main_loop_strategy: MainLoopStrategy::Split(4),
                ..Default::default()
            },
            PredefinedCmmaConfig::SplitM64k16 => CmmaConfig {
                b_m: 64,
                b_k: 16,
                b_n: 64,
                main_loop_strategy: MainLoopStrategy::Split(4),
                ..Default::default()
            },
            PredefinedCmmaConfig::TilewiseInverted => CmmaConfig {
                lhs_smem_loader_strategy: SmemLoaderStrategy::Tilewise(
                    TilingOrderStrategy::ColMajor,
                ),
                rhs_smem_loader_strategy: SmemLoaderStrategy::Tilewise(
                    TilingOrderStrategy::RowMajor,
                ),
                ..Default::default()
            },
            PredefinedCmmaConfig::Continuous => CmmaConfig {
                lhs_smem_loader_strategy: SmemLoaderStrategy::Continuous(
                    TilingOrderStrategy::RowMajor,
                ),
                rhs_smem_loader_strategy: SmemLoaderStrategy::Continuous(
                    TilingOrderStrategy::ColMajor,
                ),
                ..Default::default()
            },
            PredefinedCmmaConfig::ContinuousInverted => CmmaConfig {
                lhs_smem_loader_strategy: SmemLoaderStrategy::Continuous(
                    TilingOrderStrategy::ColMajor,
                ),
                rhs_smem_loader_strategy: SmemLoaderStrategy::Continuous(
                    TilingOrderStrategy::RowMajor,
                ),
                ..Default::default()
            },
            PredefinedCmmaConfig::LargeSmem => CmmaConfig {
                write_out_strategy: WriteOutStrategy::LargeSmem,
                ..Default::default()
            },
            PredefinedCmmaConfig::RowMajorRasterization => CmmaConfig {
                rasterization_strategy: RasterizationStrategy::RowMajor,
                ..Default::default()
            },
            PredefinedCmmaConfig::SwizzleRasterization => CmmaConfig {
                rasterization_strategy: RasterizationStrategy::Swizzle,
                ..Default::default()
            },
            PredefinedCmmaConfig::AccumulatorsFirstNoReuse => CmmaConfig {
                compute_loop_order_strategy: ComputeLoopOrderStrategy::AllAccumulatorsFirst(false),
                ..Default::default()
            },
            PredefinedCmmaConfig::BuffersFirst => CmmaConfig {
                compute_loop_order_strategy: ComputeLoopOrderStrategy::AllBuffersFirst,
                ..Default::default()
            },
            PredefinedCmmaConfig::M32K16N64 => CmmaConfig {
                b_m: 32,
                b_k: 16,
                b_n: 64,
                rhs_smem_loader_strategy: SmemLoaderStrategy::Continuous(
                    TilingOrderStrategy::ColMajor,
                ),
                ..Default::default()
            },
            PredefinedCmmaConfig::M16K32N64 => CmmaConfig {
                b_m: 16,
                b_k: 32,
                b_n: 64,
                rhs_smem_loader_strategy: SmemLoaderStrategy::Continuous(
                    TilingOrderStrategy::ColMajor,
                ),
                ..Default::default()
            },
        }
    }
}
